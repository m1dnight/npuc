// npuc-conv: PORT-32 N3a step 4 (dossier/npu-compiler.md section 6), the first compile. Builds a one-convolution
// NNNetwork with the layouts of npuc.h, as nn-eden-drv's NNNetworkBuilder does for a quantized NNAPI CONV_2D, and
// hands it to Samsung's NPUC (libnpuc_*.so from the phone's own vendor partition) in a plain Bionic process.
// Our own code; no vendor byte. Run 1 on the phone (2026-11-21 105104Z): NPUC built the graph and crashed in
// NNGraphNodeBiasONFiller::setBias on a null bias pointer (npuc.h had the kernel/bias slots at +0xf0/+0x110; they
// are +0x100/+0x120). A fault now prints its PC/LR/frames as library+offset (crash locator below).
//
//   linker64 ./npuc-conv [--k 3] [--cin 8] [--cout 8] [--hw 16] [--pad -1 (= k/2: SAME; 0: VALID)] [--soc 2] [--quant 1]
//                        [--kernel-layout oihw|ohwi] [--seed 1] [--act none|relu] [--out conv.ncp]
//   (LD_LIBRARY_PATH = the run directory; NPUC_LOG_LEVEL, NPUC_FRONTEND_NET_INFO_DUMP, NPUC_GOLDEN... are read by
//   NPUC itself)
// Output: the NCP buffer written to --out and the conv's parameters (weights, bias, quantisation) to the same name with
// .json (both announced as "FILE <name>" lines for run-n3.sh to fetch), its header words checked (magic 0x0C0FFEE0, version 22, magic2
// 0xC0DEC0DE at +0x7c), "npuc-conv: PASS" or the step that failed. Exit 0 PASS, 2 compile error, 3 bad NCP.
//
// The input is uint8 asymmetric (NNAPI TENSOR_QUANT8_ASYMM, as the HAL feeds NPUC), seeded weights and bias; the
// quantisation parameters are printed so that a numpy reference can reproduce the convolution.
#include "npuc.h"

#include <cinttypes>
#include <csignal>
#include <dlfcn.h>
#include <ucontext.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace NPUC;

// liblog replacement (NPUC imports __android_log_print unversioned; nothing in its NEEDED list provides it)
extern "C" __attribute__((visibility("default"))) int __android_log_print(int prio, const char *tag, const char *fmt, ...)
{
	va_list ap;
	fprintf(stderr, "alog %d/%s: ", prio, tag ? tag : "-");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	if (!*fmt || fmt[strlen(fmt) - 1] != '\n')
		fputc('\n', stderr);
	return 1;
}

// The exported NPUC entry points, bound at load time (the harness links against the vendor libraries' symbol
// tables only; LD_LIBRARY_PATH finds them at run time).
extern "C++" {
void npuc_NNLayer_C2(NNLayer *, std::string, std::string, std::vector<std::string>, std::vector<std::string>,
                     std::shared_ptr<LayerAttr>)
	__asm__("_ZN4NPUC7NNLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_"
	        "NS5_IS7_EEEESA_NS1_10shared_ptrINS_9LayerAttrEEE");
void npuc_NNConvLayer_C2(NNConvLayer *, std::string, std::string, std::vector<std::string>, std::vector<std::string>,
                         std::shared_ptr<ConvAttr>)
	__asm__("_ZN4NPUC11NNConvLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vector"
	        "IS7_NS5_IS7_EEEESA_NS1_10shared_ptrINS_8ConvAttrEEE");
int npuc_compile(NPUCompiler *, NNNetwork *, __NCPBuffer *)
	__asm__("_ZN4NPUC11NPUCompiler7compileEPNS_9NNNetworkEPNS_11__NCPBufferE");
void *npuc_GetInstance() __asm__("_ZN4NPUC14CompileOptions11GetInstanceEv");
}

// ---- crash locator: a fault prints its PC, LR and the frame-pointer chain as library+offset (dladdr), plus the
// address in Ghidra's view of that library (+0x100000), then the default action ends the process (exit 139 for
// SIGSEGV) -----------------------------------------------------------------------------------------------------
static const char *step = "start";

static void put(const char *s) { ssize_t r = write(2, s, strlen(s)); (void)r; }

static void where(const char *what, uintptr_t a)
{
	char b[400];
	Dl_info di;
	memset(&di, 0, sizeof di);
	if (a && dladdr((void *)a, &di) && di.dli_fname) {
		const char *n = strrchr(di.dli_fname, '/');
		uintptr_t off = a - (uintptr_t)di.dli_fbase;
		snprintf(b, sizeof b, "crash: %-3s 0x%012lx = %s+0x%lx (Ghidra 0x%lx)%s%s\n", what, (unsigned long)a,
		         n ? n + 1 : di.dli_fname, (unsigned long)off, (unsigned long)(off + 0x100000),
		         di.dli_sname ? " in " : "", di.dli_sname ? di.dli_sname : "");
	} else
		snprintf(b, sizeof b, "crash: %-3s 0x%012lx (no library)\n", what, (unsigned long)a);
	put(b);
}

static void on_fault(int sig, siginfo_t *si, void *ucv)
{
	char b[200];
	auto *uc = static_cast<ucontext_t *>(ucv);
	snprintf(b, sizeof b, "crash: signal %d code %d fault addr %p during step \"%s\"\n", sig, si->si_code, si->si_addr,
	         step);
	put(b);
	where("pc", (uintptr_t)uc->uc_mcontext.pc);
	where("lr", (uintptr_t)uc->uc_mcontext.regs[30]);
	// AAPCS64 frame records: x29 -> {previous x29, return address}; stop at anything that is not a plausible frame
	uintptr_t fp = (uintptr_t)uc->uc_mcontext.regs[29], sp = (uintptr_t)uc->uc_mcontext.sp;
	for (int i = 0; i < 24 && fp >= sp && fp - sp < (8u << 20) && (fp & 7) == 0; i++) {
		uintptr_t next = ((uintptr_t *)fp)[0], ret = ((uintptr_t *)fp)[1];
		if (!ret)
			break;
		snprintf(b, sizeof b, "#%d", i);
		where(b, ret);
		if (next <= fp)
			break;
		fp = next;
	}
	put("crash: end\n");
	// SA_RESETHAND restored the default action: returning re-runs the faulting instruction and the kernel ends us
}

static void install_crash_locator()
{
	static char alt[64 * 1024];
	stack_t ss;
	ss.ss_sp = alt;
	ss.ss_size = sizeof alt;
	ss.ss_flags = 0;
	sigaltstack(&ss, nullptr);
	struct sigaction sa;
	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = on_fault;
	sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
	sigemptyset(&sa.sa_mask);
	for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP})
		sigaction(sig, &sa, nullptr);
}

#define STEP(s) do { step = (s); printf("step: %s\n", step); } while (0)

static uint32_t rng_state;
static uint32_t rnd()
{
	rng_state = rng_state * 1664525u + 1013904223u;
	return rng_state >> 8;
}

int main(int argc, char **argv)
{
	int k = 3, cin = 8, cout = 8, hw = 16, pad = -1, soc = SOC_NEUS, quant = 1, seed = 1;
	bool ohwi = false;
	int act = 0;  // ConvAttr.fusedActivation: 0 none (NPUC: GRELU_BYPASS_ONDEVICE), 1 relu (GRELU_RELU_ONDEVICE)
	const char *out = "conv.ncp";
	for (int i = 1; i < argc; i++) {
		auto arg = [&](const char *n) { return !strcmp(argv[i], n) && i + 1 < argc; };
		if (arg("--k")) k = atoi(argv[++i]);
		else if (arg("--cin")) cin = atoi(argv[++i]);
		else if (arg("--cout")) cout = atoi(argv[++i]);
		else if (arg("--hw")) hw = atoi(argv[++i]);
		else if (arg("--pad")) pad = atoi(argv[++i]);
		else if (arg("--soc")) soc = atoi(argv[++i]);
		else if (arg("--quant")) quant = atoi(argv[++i]);
		else if (arg("--seed")) seed = atoi(argv[++i]);
		else if (arg("--out")) out = argv[++i];
		else if (arg("--kernel-layout")) ohwi = !strcmp(argv[++i], "ohwi");
		else if (arg("--act")) {
			const char *a = argv[++i];
			if (!strcmp(a, "none")) act = 0;
			else if (!strcmp(a, "relu")) act = 1;
			else { fprintf(stderr, "npuc-conv: --act none|relu\n"); return 1; }
		}
		else {
			fprintf(stderr, "npuc-conv: bad argument %s\n", argv[i]);
			return 1;
		}
	}
	if (pad < 0)
		pad = k / 2;
	// NPUC 1.36.1.a's NNConvLayer::computeOutputShape (frontend 0x97594) knows two pad types, "SAME" and "VALID";
	// "EXPLICIT" (the HAL's 10-input form, which nn-eden-drv's own inline copy handles) logs "pad type error!" and gave a
	// VALID-sized 14x14 output while the backend padded by 1 (run 105609Z): so stride-1 "same" padding is "SAME".
	const char *padType = pad == 0 ? "VALID" : (k % 2 == 1 && pad == k / 2) ? "SAME" : nullptr;
	if (!padType) {
		fprintf(stderr, "npuc-conv: pad %d with kernel %d: NPUC takes SAME (pad k/2, odd k) or VALID (pad 0) only\n", pad, k);
		return 1;
	}
	setvbuf(stdout, nullptr, _IOLBF, 0);
	install_crash_locator();
	rng_state = (uint32_t)seed;
	printf("npuc-conv: conv %dx%d, %d -> %d channels, input 1x%dx%dx%d (NCHW), pad %d, stride 1, soc %d, quant %d, "
	       "kernel layout %s\n", k, k, cin, cout, cin, hw, hw, pad, soc, quant, ohwi ? "OHWI" : "OIHW");

	STEP("CompileOptions::GetInstance");
	void *opts = npuc_GetInstance();
	printf("CompileOptions::GetInstance() = %p\n", opts);

	const std::vector<int32_t> inShape = {1, cin, hw, hw};
	// quantisation (uint8 asymmetric): real = scale * (q - zp)
	const float inS = 1.0f / 128, wS = 1.0f / 256, outS = 1.0f / 16;
	const int32_t inZ = 128, wZ = 128, outZ = 128;

	// 1. the input layer (nn-eden-drv's createDummyInputLayer: type "Input", bottom "ROOT", top = its name)
	auto ia = std::make_shared<InputAttr>();
	for (int i = 0; i < 4; i++)
		ia->shape[i] = inShape[i];
	auto *in = static_cast<NNLayer *>(::operator new(sizeof(NNLayer)));
	STEP("NNLayer C2 (input layer)");
	npuc_NNLayer_C2(in, "data0", "Input", {"ROOT"}, {"data0"}, ia);
	std::shared_ptr<NNLayer> inL(in);
	inL->shapes["data0"] = inShape;

	// 2. the convolution's attributes (loadConvAttrForConvolutionLayer)
	std::shared_ptr<ConvAttr> ca(new ConvAttr());
	ca->attrType = "Conv2D";
	ca->pad = {pad, pad, pad, pad};
	ca->stride = {1, 1, 1, 1};
	ca->dilation = {1, 1, 1, 1};
	ca->padType = padType;
	ca->group = 1;
	ca->numInput = cin;
	ca->numOutput = cout;
	ca->kernel = {k, k};
	ca->kernelName = "conv1_weight_1";
	ca->biasName = "conv1_bias_2";
	ca->hasBias = true;
	auto kd = std::make_shared<std::vector<uint8_t>>((size_t)cout * cin * k * k);
	for (auto &v : *kd)
		v = (uint8_t)(wZ - 24 + rnd() % 49);  // weights in [-24, 24] around the zero point
	auto bd = std::make_shared<std::vector<int32_t>>((size_t)cout);
	for (auto &v : *bd)
		v = (int32_t)(rnd() % 2001) - 1000;
	ca->kernelData = kd;
	ca->biasData = bd;
	ca->inScale = {inS};
	ca->inZp = {inZ};
	ca->outScale = {outS};
	ca->outZp = {outZ};
	ca->wScale = {wS};
	ca->wZp = {wZ};
	ca->bScale = {inS * wS};
	ca->bZp = {0};
	ca->fusedActivation = act;
	(void)ohwi;  // the bytes are generated, not converted: the flag only labels which layout the reference assumes

	auto *cv = static_cast<NNConvLayer *>(::operator new(sizeof(NNConvLayer)));
	STEP("NNConvLayer C2");
	npuc_NNConvLayer_C2(cv, "conv1", "Convolution", {"data0"}, {"conv1"}, ca);
	std::shared_ptr<NNLayer> cvL(cv);
	cvL->shapes["data0"] = inShape;
	// The HAL computes every layer's shapes while it builds it (getConvolutionLayer @0x6cf48: getInputShape, then
	// "convolutionLayer->computeOutputShape" through the vtable); NPUC itself does not (run 105420Z aborted in
	// NNLayer::getOutputShape: "tfShape.size() == 2 || tfShape.size() == 4" on the empty conv1 entry). NCHW, as the
	// HAL hands NPUC after convertToNCHW (CompileOptions +0x13c = 0: getOutputShape copies the shape as is).
	STEP("NNConvLayer::computeOutputShape (NPUC's)");
	cvL->computeOutputShape(std::vector<std::vector<int32_t>>{inShape});
	{
		const int ho = (hw + 2 * pad - k) + 1;
		const std::vector<int32_t> want = {1, cout, ho, ho};
		auto it = cvL->shapes.find("conv1");
		if (it == cvL->shapes.end() || it->second.size() != 4) {
			printf("computeOutputShape left conv1 with %zu dims: setting 1x%dx%dx%d ourselves\n",
			       it == cvL->shapes.end() ? (size_t)0 : it->second.size(), cout, ho, ho);
			cvL->shapes["conv1"] = want;
		} else if (it->second != want)
			printf("computeOutputShape: conv1 is %dx%dx%dx%d, expected 1x%dx%dx%d (kept NPUC's)\n", it->second[0],
			       it->second[1], it->second[2], it->second[3], cout, ho, ho);
		for (auto &l : {inL, cvL})
			for (auto &e : l->shapes) {
				printf("shape %s[%s] =", l->name.c_str(), e.first.c_str());
				for (int v : e.second)
					printf(" %d", v);
				printf("\n");
			}
	}

	// 3. the network (NNNetwork::appendLayers, inline in the HAL)
	NNNetwork net;
	STEP("append layers");
	if (npuc_append(net, inL) || npuc_append(net, cvL)) {
		printf("npuc-conv: FAIL (append)\n");
		return 2;
	}

	// 4. compile
	NPUCompilerConfig cfg;
	cfg.soc = soc;
	cfg.quantMode = quant;
	cfg.unk8 = 0;
	cfg.inputFormat = "NO_FORMAT";
	NPUCompiler comp{nullptr, &cfg};
	__NCPBuffer buf{nullptr, 0, 0};
	printf("ConvAttr %p: kernel data %p (%zu bytes) at +0x100, bias %p (%zu values) at +0x120\n", (void *)ca.get(),
	       (void *)kd->data(), kd->size(), (void *)bd->data(), bd->size());
	STEP("NPUCompiler::compile (NPUC runs: frontend fillers, GraphIR/LIR, backend NCP)");
	int rc = npuc_compile(&comp, &net, &buf);
	STEP("after compile");
	printf("NPUCompiler::compile = %d, addr %p, size %d\n", rc, buf.addr, buf.size);
	{
		auto it = cvL->shapes.find("conv1");
		if (it != cvL->shapes.end())
			printf("conv1 output shape: %d x %d x %d x %d\n", it->second.size() > 0 ? it->second[0] : -1,
			       it->second.size() > 1 ? it->second[1] : -1, it->second.size() > 2 ? it->second[2] : -1,
			       it->second.size() > 3 ? it->second[3] : -1);
	}
	if (rc != 0 || !buf.addr || buf.size < 0x80) {
		printf("npuc-conv: FAIL (compile)\n");
		return 2;
	}
	FILE *f = fopen(out, "wb");
	if (!f || fwrite(buf.addr, 1, (size_t)buf.size, f) != (size_t)buf.size || fclose(f) != 0) {
		printf("npuc-conv: FAIL (cannot write %s)\n", out);
		return 2;
	}
	uint32_t w[32];
	memcpy(w, buf.addr, sizeof w);
	printf("NCP %s: %d bytes, magic1 0x%08" PRIx32 " version %" PRIu32 " magic2 0x%08" PRIx32 "\n", out, buf.size,
	       w[0], w[1], w[31]);
	printf("quant: in scale %.9g zp %d, weight scale %.9g zp %d, bias scale %.9g, out scale %.9g zp %d, seed %d\n",
	       (double)inS, inZ, (double)wS, wZ, (double)(inS * wS), (double)outS, outZ, seed);
	if (w[0] != 0x0C0FFEE0u || w[1] != 22 || w[31] != 0xC0DEC0DEu) {
		printf("npuc-conv: FAIL (not an NCP v22 header)\n");
		return 3;
	}
	// the parameters the CPU reference needs (kernel/test-npu-n3/conv-ref.py): written next to the NCP
	std::string pj = std::string(out);
	pj = (pj.size() > 4 && pj.compare(pj.size() - 4, 4, ".ncp") == 0 ? pj.substr(0, pj.size() - 4) : pj) + ".json";
	FILE *j = fopen(pj.c_str(), "w");
	if (!j) {
		printf("npuc-conv: FAIL (cannot write %s)\n", pj.c_str());
		return 2;
	}
	auto oit = cvL->shapes.find("conv1");
	fprintf(j, "{\"ncp\": \"%s\", \"ncp_size\": %d, \"k\": %d, \"cin\": %d, \"cout\": %d, \"hw\": %d, \"pad\": %d, "
	           "\"pad_type\": \"%s\", \"stride\": 1, \"soc\": %d, \"quant_mode\": %d, \"seed\": %d, \"act\": \"%s\",\n", out,
	        buf.size, k, cin, cout, hw, pad, padType, soc, quant, seed, act ? "relu" : "none");
	fprintf(j, " \"in_scale\": %.9g, \"in_zp\": %d, \"w_scale\": %.9g, \"w_zp\": %d, \"b_scale\": %.9g, \"out_scale\": %.9g, "
	           "\"out_zp\": %d,\n", (double)inS, inZ, (double)wS, wZ, (double)(inS * wS), (double)outS, outZ);
	fprintf(j, " \"in_shape\": [1, %d, %d, %d], \"out_shape\": [", cin, hw, hw);
	if (oit != cvL->shapes.end())
		for (size_t i = 0; i < oit->second.size(); i++)
			fprintf(j, "%s%d", i ? ", " : "", oit->second[i]);
	fprintf(j, "],\n \"weights_layout\": \"%s\", \"weights_u8\": [", ohwi ? "OHWI" : "OIHW");
	for (size_t i = 0; i < kd->size(); i++)
		fprintf(j, "%s%u", i ? "," : "", (*kd)[i]);
	fprintf(j, "],\n \"bias_i32\": [");
	for (size_t i = 0; i < bd->size(); i++)
		fprintf(j, "%s%d", i ? "," : "", (*bd)[i]);
	fprintf(j, "]}\n");
	if (fclose(j) != 0) {
		printf("npuc-conv: FAIL (cannot write %s)\n", pj.c_str());
		return 2;
	}
	printf("FILE %s\nFILE %s\n", out, pj.c_str());
	printf("npuc-conv: PASS\n");
	return 0;
}
