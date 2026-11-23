// npuc-net: PORT-32, a network of our own compiled by Samsung's NPUC on the phone (npu/README.md). npuc-conv
// generalised: reads a chain of convolution and pooling layers from a text file written on the host by npu/npunet.py
// ("npuc-net 1" format below), builds the NNNetwork the way nn-eden-drv's NNNetworkBuilder does (getConvolutionLayer,
// getPoolLayer: NPUC's own layer constructors, each layer's shapes computed by NPUC's computeOutputShape as the HAL
// does) and calls NPUCompiler::compile. Our own code; no vendor byte (it links against the vendor libraries' symbol
// tables only).
//
//   linker64 ./npuc-net --net net.npuc [--out net.ncp] [--intermediate <layer>]
//   (LD_LIBRARY_PATH = the run directory; NPUC_LOG_LEVEL, NPUC_GOLDEN... are read by NPUC itself)
//   --intermediate <layer>: NPUC's one way to output a blob that a layer also reads (CompileOptions +0xb8, compared
//   with each node's layer name by NNGraphCgObject::isIntermediateOutput; an off-device knob the HAL never sets): a
//   probe, not the HAL's path (docs/format/frame.md 12)
//
// Input, whitespace-separated tokens (one layer per line by convention; lines starting with # are comments):
//   npuc-net 1
//   soc <2>  quant <1|3>
//   input <name> <C> <H> <W>
//   conv <name> <input> <cout> <kh> <kw> <sh> <sw> <SAME|VALID> <act 0|1>
//        <in scale f32 hex> <in zp> <w scale f32 hex> <w zp> <out scale f32 hex> <out zp> <weights: OIHW uint8, hex> <nbias> <bias int32...>
//   pool <name> <input> <MaxPool|AvgPool> <kh> <kw> <sh> <sw> <SAME|VALID> <act 0|1> <in scale hex> <in zp> <out scale hex> <out zp>
//   end
// Scales travel as float32 bit patterns (0x3c000000 = 1/128): NPUC truncates the multiplier it derives from them
// (ScaleONFiller::setScale), so the host's CPU reference must see the very same floats.
//
// "npuc-net 2" files may also hold these lines (dossier/npuc-harness-ops.md; layers NPUC's graph layer has but the
// stock HAL never hands it). A layer's inputs are any earlier blobs (the input or a layer), so graphs are allowed:
//   eltwise <name> <in1> <in2> Add <in1 scale hex> <in1 zp> <in2 scale hex> <in2 zp> <out scale hex> <out zp>
//        two different blobs of the same C x H x W; Caffe type "Eltwise". NPUC's Eltwise adds by powers of two only:
//        every scale must be 2^-n (n = its fractional length, given to NPUC through the hook below) and the three zero
//        points equal (no zero-point term: quant 3 with zero points 0 is the meaningful case)
//   concat <name> 1 <n> <in1> ... <in_n> <out scale hex> <out zp> [mixed]
//        channel concatenation (NCHW axis 1, the only one NPUC does), H and W equal; Caffe type "Concat",
//        nn-eden-drv's ConcatAttr; NPUC places each input as a channel slice, in this order, without rescaling.
//        "mixed": the inputs keep their own quantisations (the bytes are placed as they are; NPUC reads no concat
//        quantisation on this path) and the out scale and zero point are what its readers declare: npunet's lowering
//        of an ADD with any scales, a 1x1 conv over the concatenation that weighs each input's channels on its own
//        and takes the zero-point difference into its bias (docs/format/add.md)
//   relu <name> <input> <ReLU|ReLU6>
//        a standalone activation layer (Caffe type "ReLU"/"ReLU6"; NPUC makes it a 1x1 convolution)
//   dwconv <name> <input> <stock|dwopt> <kh> <kw> <sh> <sw> <SAME|VALID> <act 0|1>
//        <in scale hex> <in zp> <w scale hex> <w zp> <out scale hex> <out zp> <weights: C x kh x kw uint8, hex> <nbias> <bias...>
//        a depthwise convolution, multiplier 1 (C in = C out = group, C >= 2; docs/format/depthwise.md): a Caffe
//        "Convolution" whose ConvAttr has group = numInput = numOutput = C (as nn-eden-drv's ConvAttr::setKernelInfo
//        sets it for a non-"Conv2D" attribute). Its kernel goes to NPUC as NPUC's own off-device KernelFiller expands a
//        depthwise kernel: dense C x C x kh x kw, the channel's taps on the diagonal (o == i), the weight zero point
//        (a real 0) elsewhere. The weights here are per channel, [c][ky][kx]. Mode: "stock" is what NPUC 1.36.1.a does
//        on the device (NNGraphAttribute::DwOptSupported is false there: it asks for the framework "scaffe" and the
//        on-device frontend is "stensorflow"); "dwopt" makes DwOptSupported see "scaffe" for this layer (the hook
//        below), so NPUC's own remaining tests decide whether its depthwise path (group = C) is taken
//   output <n> <blob1> ... <blob_n>
//        the network's outputs, the way the HAL hands them to NPUC: nothing marks an output in the NNNetwork; NPUC's
//        frontend makes every node without successors an output (NNGraphCgObject::appendNewOutData, an UNFLATTEN op
//        and an output data node each, in its node order; docs/format/frame.md 12). So this line only declares them and
//        is checked against that rule: every blob read by no layer must be listed, and a listed blob must be read by
//        no layer (unless it is the --intermediate one). Without the line: as before (a note for each unread blob)
// In an "npuc-net 2" file every reader must declare the scale and zero point its input blob already has (NPUC does not
// rescale between layers on this path); "npuc-net 1" files are read exactly as before (the new lines refused there).
//
// Per-channel requantisation (docs/format/perchannel.md), in either version, after the conv/dwconv line they qualify:
//   wscales <layer> <n> <w scale f32 hex> x n
//        the conv's weight scale as a per-channel vector of n = C out floats (ConvAttr.wScale, and bScale = in x w per
//        channel), what the HAL would hand NPUC for a per-channel operand if it read one (it reads Operand.scale only)
//   slopes <layer> <n> <slope> x n <shift> x n
//        a per-channel multiplier: after NPUC's own NNGraphNodeScaleONFiller::setScale (which derives one slope and one
//        shift from wScale[0] and stores them C out times in the node's per-channel vectors), the hook below writes
//        these n = C out slopes (u16) and shifts (8..15) into those same vectors
// Neither line exists in NPUC's input model: they are this harness's way of asking what NPUC's backend does with
// per-channel values its frontend never produces.
//
// Output: the NCP written to --out and <out>.json (what NPUC made of each layer: its output shape and pads, read back
// after computeOutputShape; the NCP's size), both announced as "FILE <name>" for run-n3.sh to fetch; "npuc-net: PASS"
// or the step that failed. Exit 0 PASS, 1 bad input, 2 compile error, 3 not an NCP.
#include "npuc.h"

#include <cctype>
#include <cinttypes>
#include <csignal>
#include <dlfcn.h>
#include <ucontext.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

extern "C++" {
void npuc_NNLayer_C2(NNLayer *, std::string, std::string, std::vector<std::string>, std::vector<std::string>,
                     std::shared_ptr<LayerAttr>)
	__asm__("_ZN4NPUC7NNLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_"
	        "NS5_IS7_EEEESA_NS1_10shared_ptrINS_9LayerAttrEEE");
void npuc_NNConvLayer_C2(NNConvLayer *, std::string, std::string, std::vector<std::string>, std::vector<std::string>,
                         std::shared_ptr<ConvAttr>)
	__asm__("_ZN4NPUC11NNConvLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vector"
	        "IS7_NS5_IS7_EEEESA_NS1_10shared_ptrINS_8ConvAttrEEE");
void npuc_NNPoolLayer_C2(NNPoolLayer *, std::string, std::string, std::vector<std::string>, std::vector<std::string>,
                         std::shared_ptr<PoolAttr>)
	__asm__("_ZN4NPUC11NNPoolLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vector"
	        "IS7_NS5_IS7_EEEESA_NS1_10shared_ptrINS_8PoolAttrEEE");
int npuc_compile(NPUCompiler *, NNNetwork *, __NCPBuffer *)
	__asm__("_ZN4NPUC11NPUCompiler7compileEPNS_9NNNetworkEPNS_11__NCPBufferE");
// weak: the host test's stub NPUC (a static build) does not define it; NPUC's libnpuc_frontend exports it
void npuc_mergeConcatLayer(NNNetwork *) __asm__("_ZN4NPUC9NNNetwork16mergeConcatLayerEv") __attribute__((weak));
void *npuc_GetInstance() __asm__("_ZN4NPUC14CompileOptions11GetInstanceEv");
}

// ---- Eltwise: the one thing NPUC's on-device chain leaves out (dossier/npuc-harness-ops.md) ---------------------
// NPUC's Eltwise op aligns its two inputs by power-of-two exponents (fractional lengths) that only the off-device
// NNGraphNodeMultipleInputFiller sets (node +0xe8); on the on-device chain the slot stays empty and the LIR generator
// dereferences it (SIGSEGV in EltwiseNode2020::buildMetaDataDram). This executable defines NPUC's exported
// NNGraphCgObject::addElemtWiseGraphNode, which addGraphNode calls through the PLT (so Bionic binds that call here,
// the executable being first in the lookup scope): for an Eltwise node of ours it fills the slot with the fractional
// lengths read from the net file, then runs NPUC's own function (RTLD_NEXT). Networks without Eltwise never get here.
struct EltwiseFL {
	int fl1, fl2, flo;
};
static std::map<std::string, EltwiseFL> g_eltwise;  // layer name -> fractional lengths (scale = 2^-fl)

#if defined(__aarch64__)  // (the host test's x86_64 build with a stub NPUC has neither NPUC's vtable nor libdl)
extern "C++" __attribute__((visibility("default"))) void npuc_hook_addElemtWiseGraphNode(void *self, void *node)
	__asm__("_ZN4NPUC15NNGraphCgObject21addElemtWiseGraphNodeEPNS_11NNGraphNodeE");
void npuc_hook_addElemtWiseGraphNode(void *self, void *node)
{
	using fn_t = void (*)(void *, void *);
	static fn_t orig = reinterpret_cast<fn_t>(dlsym(RTLD_NEXT, kSymAddElemtWise));
	auto *n = static_cast<char *>(node);
	NNLayer *l = *reinterpret_cast<NNLayer **>(n + NNGraphNodeOff::layer);
	auto it = l ? g_eltwise.find(l->name) : g_eltwise.end();
	if (it != g_eltwise.end() && !l->outputs.empty()) {
		auto &slot = *reinterpret_cast<std::shared_ptr<NPUMultipleInputMemoryData> *>(n + NNGraphNodeOff::multipleInput);
		const auto &os = l->shapes[l->outputs[0]];
		size_t c = os.size() == 4 ? (size_t)os[1] : 0;
		if (slot)
			printf("eltwise %s: NPUC already set its multiple-input data; left as is\n", l->name.c_str());
		else {
			std::shared_ptr<NPUMultipleInputMemoryData> d(new NPUMultipleInputMemoryData());
			d->fl[0].assign(c, (uint8_t)it->second.fl1);
			d->fl[1].assign(c, (uint8_t)it->second.fl2);
			d->fl[8].assign(c, (uint8_t)it->second.flo);
			slot = d;
			printf("eltwise %s: multiple-input data set (node type 0x%x): %zu channels, fractional lengths in1 %d in2 %d "
			       "out %d\n", l->name.c_str(), *reinterpret_cast<int32_t *>(n + NNGraphNodeOff::type), c,
			       it->second.fl1, it->second.fl2, it->second.flo);
		}
	}
	if (!orig) {
		printf("npuc-net: FAIL (no NPUC addElemtWiseGraphNode behind the hook)\n");
		_exit(2);
	}
	orig(self, node);
}
#endif

// ---- Depthwise: NPUC's DW path, behind a test the on-device frontend always fails (docs/format/depthwise.md) ------
// NNGraphAttribute::DwOptSupported() (libnpuc_frontend 0x5333c) wants group > 1, a Convolution node, a kernel of
// C x C (C a multiple of 16) and the attribute's framework string (+0x18) equal to "scaffe"; NNGraphAttribute's
// constructor sets "stensorflow", so on the device it is always false. addConvolutionGraphNode (frontend) and
// PolicyMaker::makeCompressedKernel (template) call it through their PLTs, so Bionic binds those calls to this
// definition (the executable first in the lookup scope). For a "dwconv ... dwopt" layer it lets NPUC's own function
// run with the framework string set to "scaffe", then puts "stensorflow" back; for every other node it is NPUC's
// function unchanged. Networks without such a layer never change behaviour.
static std::map<std::string, int> g_dwopt;  // layer name -> number of DwOptSupported calls answered with "scaffe"

#if defined(__aarch64__)
extern "C++" __attribute__((visibility("default"))) bool npuc_hook_DwOptSupported(void *self)
	__asm__("_ZN4NPUC16NNGraphAttribute14DwOptSupportedEv");
bool npuc_hook_DwOptSupported(void *self)
{
	using fn_t = bool (*)(void *);
	static fn_t orig = reinterpret_cast<fn_t>(dlsym(RTLD_NEXT, kSymDwOptSupported));
	if (!orig) {
		printf("npuc-net: FAIL (no NPUC DwOptSupported behind the hook)\n");
		_exit(2);
	}
	auto *a = static_cast<char *>(self);
	NNLayer *l = *reinterpret_cast<NNLayer **>(a + NNGraphAttributeOff::layer);
	auto it = l ? g_dwopt.find(l->name) : g_dwopt.end();
	if (it == g_dwopt.end())
		return orig(self);
	auto &fw = *reinterpret_cast<std::string *>(a + NNGraphAttributeOff::framework);
	std::string saved = fw;
	fw = "scaffe";
	bool r = orig(self);
	fw = saved;
	if (it->second++ == 0)
		printf("dwconv %s: NNGraphAttribute::DwOptSupported with framework \"scaffe\" (was \"%s\") = %d\n",
		       l->name.c_str(), saved.c_str(), (int)r);
	return r;
}
#endif

// ---- Per-channel multipliers: NNGraphNodeScaleONFiller::setScale (libnpuc_frontend 0x9da64) -------------------------
// NPUC's on-device scale filler reads ConvAttr inScale.at(0), outScale.at(0) and wScale.at(0) (attr +0x130, +0x160,
// +0x190), derives one slope (fcvtas(m x 256), m doubled into [0.5, 1)) and one shift (doublings + 8), and pushes them
// numOutput (attr +0x58) times into a new NPUSlopeMemoryData (node +0x108: object, +0x110: control block; its
// vector<uint32_t> at object +8) and a shift object (node +0x118 / +0x120; vector<uint8_t> at object +8). Its run()
// calls setScale through the PLT (@0x9d908: b setScale@plt), so Bionic binds the call to this definition. For a layer
// with a "slopes" line it runs NPUC's own function, then overwrites the two vectors element by element; every other
// node is NPUC's function unchanged.
struct PerChannel {
	std::vector<uint32_t> slope;
	std::vector<uint8_t> shift;
	int calls = 0;
};
static std::map<std::string, PerChannel> g_slopes;  // layer name -> per-channel slope and shift

#if defined(__aarch64__)
constexpr const char *kSymSetScale = "_ZN4NPUC24NNGraphNodeScaleONFiller8setScaleERNS_11NNGraphNodeE";
extern "C++" __attribute__((visibility("default"))) void npuc_hook_setScale(void *self, void *node)
	__asm__("_ZN4NPUC24NNGraphNodeScaleONFiller8setScaleERNS_11NNGraphNodeE");
void npuc_hook_setScale(void *self, void *node)
{
	using fn_t = void (*)(void *, void *);
	static fn_t orig = reinterpret_cast<fn_t>(dlsym(RTLD_NEXT, kSymSetScale));
	if (!orig) {
		printf("npuc-net: FAIL (no NPUC setScale behind the hook)\n");
		_exit(2);
	}
	orig(self, node);
	auto *n = static_cast<char *>(node);
	NNLayer *l = *reinterpret_cast<NNLayer **>(n + NNGraphNodeOff::layer);
	auto it = l ? g_slopes.find(l->name) : g_slopes.end();
	if (it == g_slopes.end())
		return;
	char *so = *reinterpret_cast<char **>(n + 0x108), *ho = *reinterpret_cast<char **>(n + 0x118);
	if (!so || !ho) {
		printf("npuc-net: FAIL (setScale left no slope/shift data for %s)\n", l->name.c_str());
		_exit(2);
	}
	auto &sv = *reinterpret_cast<std::vector<uint32_t> *>(so + 8);
	auto &hv = *reinterpret_cast<std::vector<uint8_t> *>(ho + 8);
	PerChannel &pc = it->second;
	if (sv.size() != pc.slope.size() || hv.size() != pc.shift.size()) {
		printf("npuc-net: FAIL (%s: NPUC's per-channel vectors hold %zu slopes / %zu shifts, the file %zu)\n",
		       l->name.c_str(), sv.size(), hv.size(), pc.slope.size());
		_exit(2);
	}
	printf("slopes %s: NPUC's setScale gave slope %u shift %u x %zu; replaced by the file's per-channel values\n",
	       l->name.c_str(), sv.empty() ? 0u : sv[0], hv.empty() ? 0u : (unsigned)hv[0], sv.size());
	for (size_t c = 0; c < sv.size(); c++)
		sv[c] = pc.slope[c], hv[c] = pc.shift[c];
	pc.calls++;
}
#endif

// ---- crash locator (as npuc-conv): a fault prints PC, LR and the frame-pointer chain as library+offset ---------
static const char *step = "start";
static std::string stepbuf;

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
	char b[300];
	auto *uc = static_cast<ucontext_t *>(ucv);
	snprintf(b, sizeof b, "crash: signal %d code %d fault addr %p during step \"%s\"\n", sig, si->si_code, si->si_addr,
	         step);
	put(b);
#if defined(__aarch64__)  // (the host test builds npuc-net for x86_64 with a stub NPUC: no frame walk there)
	where("pc", (uintptr_t)uc->uc_mcontext.pc);
	where("lr", (uintptr_t)uc->uc_mcontext.regs[30]);
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
#else
	(void)uc;
	(void)where;
#endif
	put("crash: end\n");
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

static void STEP(const std::string &s)
{
	stepbuf = s;
	step = stepbuf.c_str();
	printf("step: %s\n", step);
}

// ---- the input file ------------------------------------------------------------------------------------------
struct Tok {
	std::vector<std::string> t;
	size_t i = 0;
	bool bad = false;
	std::string why;
	bool more() const { return i < t.size(); }
	std::string str()
	{
		if (i >= t.size()) {
			fail("unexpected end of file");
			return "";
		}
		return t[i++];
	}
	long num(long lo, long hi, const char *what)
	{
		std::string s = str();
		char *e = nullptr;
		long v = strtol(s.c_str(), &e, 0);
		if (s.empty() || *e || v < lo || v > hi)
			fail(std::string(what) + " '" + s + "' out of range");
		return v;
	}
	float f32hex(const char *what)
	{
		std::string s = str();
		char *e = nullptr;
		unsigned long v = strtoul(s.c_str(), &e, 16);
		float f;
		uint32_t u = (uint32_t)v;
		memcpy(&f, &u, 4);
		if (s.size() != 10 || s.compare(0, 2, "0x") || *e || !(f > 0.0f) || f != f)
			fail(std::string(what) + " '" + s + "' is not a positive float32 bit pattern 0x........");
		return f;
	}
	void fail(const std::string &w)
	{
		if (!bad) {
			bad = true;
			why = w + " (token " + std::to_string(i) + ")";
		}
	}
};

struct Shape {
	int c = 0, h = 0, w = 0;
};

static bool read_file(const char *path, std::string &out)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return false;
	char buf[65536];
	size_t n;
	while ((n = fread(buf, 1, sizeof buf, f)) > 0)
		out.append(buf, n);
	fclose(f);
	return true;
}

static std::vector<uint8_t> unhex(const std::string &s, bool &ok)
{
	std::vector<uint8_t> v;
	ok = s.size() % 2 == 0;
	for (size_t i = 0; ok && i < s.size(); i += 2) {
		char b[3] = {s[i], s[i + 1], 0};
		char *e = nullptr;
		long x = strtol(b, &e, 16);
		ok = *e == 0 && isxdigit((unsigned char)b[0]) && isxdigit((unsigned char)b[1]);
		v.push_back((uint8_t)x);
	}
	return v;
}

static void print_shapes(const std::shared_ptr<NNLayer> &l)
{
	for (auto &e : l->shapes) {
		printf("shape %s[%s] =", l->name.c_str(), e.first.c_str());
		for (int v : e.second)
			printf(" %d", v);
		printf("\n");
	}
}

int main(int argc, char **argv)
{
	const char *netf = nullptr, *out = "net.ncp";
	std::string inter;  // --intermediate: the layer whose blob NPUC also outputs (CompileOptions +0xb8)
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--net") && i + 1 < argc)
			netf = argv[++i];
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			out = argv[++i];
		else if (!strcmp(argv[i], "--intermediate") && i + 1 < argc)
			inter = argv[++i];
		else {
			fprintf(stderr, "npuc-net: bad argument %s\n", argv[i]);
			return 1;
		}
	}
	if (!netf) {
		fprintf(stderr, "npuc-net: --net <file> is required\n");
		return 1;
	}
	setvbuf(stdout, nullptr, _IOLBF, 0);
	install_crash_locator();
	std::string text;
	if (!read_file(netf, text)) {
		printf("npuc-net: FAIL (cannot read %s)\n", netf);
		return 1;
	}
	Tok tk;
	// tokens: whitespace-separated; a line whose first character is '#' is a comment (no iostreams: NPUC's exported
	// libc++, the one C++ runtime of this process, has no istringstream for the NDK's headers)
	for (size_t i = 0, bol = 1; i < text.size();) {
		char c = text[i];
		if (bol && c == '#') {
			while (i < text.size() && text[i] != '\n')
				i++;
			continue;
		}
		if (isspace((unsigned char)c)) {
			bol = c == '\n';
			i++;
			continue;
		}
		size_t j = i;
		while (j < text.size() && !isspace((unsigned char)text[j]))
			j++;
		tk.t.push_back(text.substr(i, j - i));
		i = j;
		bol = 0;
	}
	std::string magic = tk.str(), version = tk.str();
	if (magic != "npuc-net" || (version != "1" && version != "2")) {
		printf("npuc-net: FAIL (%s is not an \"npuc-net 1\" or \"npuc-net 2\" file)\n", netf);
		return 1;
	}
	const int ver = version == "2" ? 2 : 1;
	bool hasConcat = false;
	std::vector<std::shared_ptr<LayerAttr>> attrs;  // the new kinds' attributes (kept alive until the compile is over)
	int soc = SOC_NEUS, quant = 1;
	std::map<std::string, Shape> blobs;  // blob name -> CHW shape (NPUC's, read back)
	// "npuc-net 2" only: each blob's quantisation (its producer's output scale and zero point; the input's from its first
	// reader). On this path NPUC never rescales between layers (no graph filter runs on the on-device chain, so no
	// ConcatScaleFolding; Eltwise only shifts), so a reader that declares another scale is an error in the file.
	struct BlobQ {
		float s;
		int zp;
	};
	std::map<std::string, BlobQ> bq;
	auto qin = [&](Tok &t, const std::string &layer, const std::string &blob, float s, int zp) {
		if (ver < 2)
			return;
		auto it = bq.find(blob);
		if (it == bq.end())
			bq[blob] = {s, zp};
		else if (memcmp(&it->second.s, &s, 4) != 0 || it->second.zp != zp)
			t.fail("layer " + layer + ": reads " + blob + " as scale " + std::to_string(s) + " zp " + std::to_string(zp) +
			       ", but " + blob + " is scale " + std::to_string(it->second.s) + " zp " + std::to_string(it->second.zp) +
			       " (NPUC does not rescale between layers)");
	};
	std::vector<std::shared_ptr<NNLayer>> layers;
	std::vector<std::shared_ptr<ConvAttr>> convs;  // kept alive with their data until the compile is over
	std::map<std::string, std::shared_ptr<ConvAttr>> convByName;  // for the per-channel lines
	std::vector<std::string> order;
	std::string last;
	STEP("CompileOptions::GetInstance");
	void *opts = npuc_GetInstance();
	printf("CompileOptions::GetInstance() = %p\n", opts);
	std::string jl;  // the per-layer JSON, assembled while the layers are built (no tmpfile: Bionic's wants /data/local/tmp)
	bool first = true;
	std::vector<std::string> outs;  // the "output" line's blobs (empty: none declared)
	while (!tk.bad && tk.more()) {
		std::string kw = tk.str();
		if (kw == "end")
			break;
		if (kw == "output") {
			if (ver < 2) {
				tk.fail("'output' needs an \"npuc-net 2\" file");
				break;
			}
			if (!outs.empty()) {
				tk.fail("output: one line");
				break;
			}
			long n = tk.num(1, 64, "number of outputs");
			for (long j = 0; j < n && !tk.bad; j++) {
				std::string o = tk.str();
				for (auto &x : outs)
					if (x == o)
						tk.fail("output: " + o + " twice");
				outs.push_back(o);
			}
		} else if (kw == "soc") {
			soc = (int)tk.num(0, 3, "soc");
		} else if (kw == "quant") {
			quant = (int)tk.num(1, 3, "quant");
		} else if (kw == "input") {
			std::string name = tk.str();
			Shape s;
			s.c = (int)tk.num(1, 4096, "C");
			s.h = (int)tk.num(1, 4096, "H");
			s.w = (int)tk.num(1, 4096, "W");
			if (tk.bad || !layers.empty()) {
				tk.fail("input: one, first");
				break;
			}
			auto ia = std::make_shared<InputAttr>();
			ia->shape[0] = 1, ia->shape[1] = s.c, ia->shape[2] = s.h, ia->shape[3] = s.w;
			auto *in = static_cast<NNLayer *>(::operator new(sizeof(NNLayer)));
			STEP("NNLayer C2 (input " + name + ")");
			npuc_NNLayer_C2(in, name, "Input", {"ROOT"}, {name}, ia);
			std::shared_ptr<NNLayer> l(in);
			l->shapes[name] = {1, s.c, s.h, s.w};
			if (ver >= 2) {
				// as nn-eden-drv's createDummyInputLayer: NNLayer::computeOutputShape (NPUC's) on the input layer, which
				// also records shapes["ROOT"]; NPUC's policy maker reads it when the input feeds a Concat directly
				STEP("computeOutputShape (NPUC's) for input " + name);
				l->computeOutputShape(std::vector<std::vector<int32_t>>{{1, s.c, s.h, s.w}});
			}
			layers.push_back(l);
			blobs[name] = s;
			last = name;
			printf("layer %s Input out %dx%dx%d\n", name.c_str(), s.c, s.h, s.w);
		} else if (kw == "conv" || kw == "pool" || kw == "dwconv") {
			if (kw == "dwconv" && ver < 2) {
				tk.fail("'dwconv' needs an \"npuc-net 2\" file");
				break;
			}
			std::string name = tk.str(), input = tk.str();
			auto bi = blobs.find(input);
			if (bi == blobs.end()) {
				tk.fail("layer " + name + ": unknown input " + input);
				break;
			}
			if (blobs.count(name)) {
				tk.fail("layer " + name + ": duplicate name");
				break;
			}
			Shape is = bi->second;
			const std::vector<int32_t> inShape = {1, is.c, is.h, is.w};
			std::shared_ptr<NNLayer> l;
			int cout = is.c, kh, kw_, sh, sw, act, group = 1;
			std::string padType, ptype, dwmode;
			if (kw == "conv" || kw == "dwconv") {
				const bool dw = kw == "dwconv";
				if (dw) {
					dwmode = tk.str();
					if (dwmode != "stock" && dwmode != "dwopt")
						tk.fail("dwconv " + name + ": mode " + dwmode + " (stock or dwopt)");
					if (is.c < 2)
						tk.fail("dwconv " + name + ": " + std::to_string(is.c) + " channel (NPUC's isDwConvolution wants group > 1)");
					group = is.c;
				} else
					cout = (int)tk.num(1, 4096, "cout");
				kh = (int)tk.num(1, 64, "kh"), kw_ = (int)tk.num(1, 64, "kw");
				sh = (int)tk.num(1, 16, "sh"), sw = (int)tk.num(1, 16, "sw");
				padType = tk.str();
				act = (int)tk.num(0, 1, "act");
				float inS = tk.f32hex("in scale");
				int inZ = (int)tk.num(0, 255, "in zp");
				float wS = tk.f32hex("w scale");
				int wZ = (int)tk.num(0, 255, "w zp");
				float outS = tk.f32hex("out scale");
				int outZ = (int)tk.num(0, 255, "out zp");
				bool ok;
				std::vector<uint8_t> wbytes = unhex(tk.str(), ok);
				size_t nw = dw ? (size_t)is.c * kh * kw_ : (size_t)cout * is.c * kh * kw_;
				if (!ok || wbytes.size() != nw)
					tk.fail(kw + " " + name + ": weights: want " + std::to_string(nw) + " bytes in hex, got " +
					        std::to_string(wbytes.size()));
				if (dw && !tk.bad) {
					// NPUC's off-device NNGraphNodeKernelFiller::setKernel expands a depthwise kernel to a dense
					// C x C x kh x kw one, the channel's taps where o == i and a zero weight elsewhere; the on-device
					// KernelONFiller takes numOutput x numInput x kh x kw bytes as they are. A real 0 is the weight
					// zero point (quant 3: 128, which saveKernelDataWithSubstracting128 makes 0)
					std::vector<uint8_t> dense((size_t)is.c * is.c * kh * kw_, (uint8_t)wZ);
					const size_t kk = (size_t)kh * kw_;
					for (int c = 0; c < is.c; c++)
						memcpy(&dense[((size_t)c * is.c + c) * kk], &wbytes[(size_t)c * kk], kk);
					wbytes.swap(dense);
				}
				long nb = tk.num(0, 4096, "nbias");
				if (nb != cout)
					tk.fail("conv " + name + ": " + std::to_string(nb) + " biases for " + std::to_string(cout) + " outputs");
				auto bd = std::make_shared<std::vector<int32_t>>();
				for (long j = 0; j < nb && !tk.bad; j++)
					bd->push_back((int32_t)tk.num(-2147483647L - 1, 2147483647L, "bias"));
				if (padType != "SAME" && padType != "VALID")
					tk.fail(kw + " " + name + ": pad type " + padType + " (NPUC 1.36.1.a's NNConvLayer knows SAME and VALID)");
				qin(tk, name, input, inS, inZ);
				if (tk.bad)
					break;
				if (ver >= 2)
					bq[name] = {outS, outZ};
				if (dw && dwmode == "dwopt")
					g_dwopt[name] = 0;
				// loadConvAttrForConvolutionLayer, as npuc-conv
				std::shared_ptr<ConvAttr> ca(new ConvAttr());
				// nn-eden-drv's ConvAttr::setKernelInfo: "Conv2D" -> group 1; any other attribute type is depthwise:
				// group = numOutput = numInput (NPUC itself never reads attrType on this path)
				ca->attrType = dw ? "DepthwiseConv2dNative" : "Conv2D";
				ca->pad = {0, 0, 0, 0};  // NPUC's computeOutputShape writes SAME's pads back here
				ca->stride = {1, 1, sh, sw};
				ca->dilation = {1, 1, 1, 1};
				ca->padType = padType;
				ca->group = group;
				ca->numInput = is.c;
				ca->numOutput = cout;
				ca->kernel = {kh, kw_};
				ca->kernelName = name + "_weight_1";
				ca->biasName = name + "_bias_2";
				ca->hasBias = true;
				ca->kernelData = std::make_shared<std::vector<uint8_t>>(wbytes);
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
				convs.push_back(ca);
				convByName[name] = ca;
				auto *cv = static_cast<NNConvLayer *>(::operator new(sizeof(NNConvLayer)));
				STEP("NNConvLayer C2 (" + name + ")");
				npuc_NNConvLayer_C2(cv, name, "Convolution", {input}, {name}, ca);
				l.reset(cv);
				printf("layer %s Convolution in %s %dx%dx%d, %dx%d stride %dx%d %s act %d, %d -> %d channels, "
				       "scales in %.9g zp %d w %.9g zp %d out %.9g zp %d\n", name.c_str(), input.c_str(), is.c, is.h,
				       is.w, kh, kw_, sh, sw, padType.c_str(), act, is.c, cout, (double)inS, inZ, (double)wS, wZ,
				       (double)outS, outZ);
				if (dw)
					printf("layer %s depthwise: group %d, mode %s, kernel expanded to %d x %d x %d x %d (diagonal, zero "
					       "weight %d elsewhere)\n", name.c_str(), group, dwmode.c_str(), cout, is.c, kh, kw_, wZ);
			} else {
				ptype = tk.str();
				kh = (int)tk.num(1, 64, "kh"), kw_ = (int)tk.num(1, 64, "kw");
				sh = (int)tk.num(1, 64, "sh"), sw = (int)tk.num(1, 64, "sw");
				padType = tk.str();
				act = (int)tk.num(0, 1, "act");
				float inS = tk.f32hex("in scale");
				int inZ = (int)tk.num(0, 255, "in zp");
				float outS = tk.f32hex("out scale");
				int outZ = (int)tk.num(0, 255, "out zp");
				if (ptype != "MaxPool" && ptype != "AvgPool")
					tk.fail("pool " + name + ": type " + ptype + " (MaxPool or AvgPool)");
				if (padType != "SAME" && padType != "VALID")
					tk.fail("pool " + name + ": pad type " + padType + " (SAME or VALID)");
				qin(tk, name, input, inS, inZ);
				if (tk.bad)
					break;
				if (ver >= 2)
					bq[name] = {outS, outZ};
				// loadPoolAttrForPoolLayer
				auto pa = std::make_shared<PoolAttr>();
				pa->pad = {0, 0, 0, 0};
				pa->stride = {1, 1, sh, sw};
				pa->kernel = {1, 1, kh, kw_};
				pa->padType = padType;
				pa->poolType = ptype;
				pa->inScale = {inS};
				pa->inZp = {inZ};
				pa->outScale = {outS};
				pa->outZp = {outZ};
				pa->fusedActivation = act;
				auto *pl = static_cast<NNPoolLayer *>(::operator new(sizeof(NNPoolLayer)));
				STEP("NNPoolLayer C2 (" + name + ")");
				npuc_NNPoolLayer_C2(pl, name, "Pooling", {input}, {name}, pa);
				l.reset(pl);
				printf("layer %s Pooling %s in %s %dx%dx%d, %dx%d stride %dx%d %s act %d, scales in %.9g zp %d out "
				       "%.9g zp %d\n", name.c_str(), ptype.c_str(), input.c_str(), is.c, is.h, is.w, kh, kw_, sh, sw,
				       padType.c_str(), act, (double)inS, inZ, (double)outS, outZ);
			}
			l->shapes[input] = inShape;
			// the HAL computes each layer's shapes as it builds it (getConvolutionLayer / getPoolLayer: then
			// "->computeOutputShape" through the vtable); NPUC itself does not (npu-compiler.md section 9.8)
			STEP("computeOutputShape (NPUC's) for " + name);
			l->computeOutputShape(std::vector<std::vector<int32_t>>{inShape});
			auto it = l->shapes.find(name);
			if (it == l->shapes.end() || it->second.size() != 4 || it->second[0] != 1 || it->second[1] != cout ||
			    it->second[2] < 1 || it->second[3] < 1) {
				print_shapes(l);
				tk.fail("layer " + name + ": NPUC's computeOutputShape gave no 1 x " + std::to_string(cout) +
				        " x H x W output (pad type " + padType + ")");
				break;
			}
			Shape os;
			os.c = it->second[1], os.h = it->second[2], os.w = it->second[3];
			blobs[name] = os;
			std::vector<int32_t> pads = kw != "pool" ? convs.back()->pad
			                                         : std::static_pointer_cast<PoolAttr>(l->attr)->pad;
			while (pads.size() < 4)
				pads.push_back(-1);
			printf("layer %s out %dx%dx%d pads %d %d %d %d (top bottom left right, NPUC's)\n", name.c_str(), os.c,
			       os.h, os.w, pads[0], pads[1], pads[2], pads[3]);
			print_shapes(l);
			char lb[512];
			snprintf(lb, sizeof lb, "%s{\"name\": \"%s\", \"type\": \"%s\", \"input\": \"%s\", \"out_shape\": [1, %d, %d, %d], "
			         "\"pads\": [%d, %d, %d, %d]", first ? "" : ",\n  ", name.c_str(),
			         kw == "pool" ? "Pooling" : "Convolution", input.c_str(), os.c, os.h, os.w, pads[0], pads[1], pads[2],
			         pads[3]);
			jl += lb;
			if (kw == "dwconv") {
				snprintf(lb, sizeof lb, ", \"group\": %d, \"dw_mode\": \"%s\"", group, dwmode.c_str());
				jl += lb;
			}
			jl += "}";
			first = false;
			layers.push_back(l);
			last = name;
		} else if (kw == "eltwise" || kw == "concat" || kw == "relu") {
			if (ver < 2) {
				tk.fail("'" + kw + "' needs an \"npuc-net 2\" file");
				break;
			}
			std::string name = tk.str(), type;
			std::vector<std::string> ins;
			std::shared_ptr<LayerAttr> attr;
			Shape os;
			char extra[256] = "";
			if (kw == "eltwise") {
				ins.push_back(tk.str());
				ins.push_back(tk.str());
				if (ins[0] == ins[1])
					tk.fail("eltwise " + name + ": both inputs are " + ins[0] +
					        " (NPUC's policy maker asserts on it: PGraph::clone \"this->equals(graph)\")");
				std::string op = tk.str();
				float s1 = tk.f32hex("in1 scale");
				int z1 = (int)tk.num(0, 255, "in1 zp");
				float s2 = tk.f32hex("in2 scale");
				int z2 = (int)tk.num(0, 255, "in2 zp");
				float so = tk.f32hex("out scale");
				int zo = (int)tk.num(0, 255, "out zp");
				// getEltwiseParam knows Add, Multiply and Maximum, but NPUC 1.36.1.a's backend implements Add only
				// (Multiply: "Unsupported eltwise op!" in NcpEltwiseOp::Execute; Maximum: an assert in
				// EltwiseNode2020::getEltwiseOpcode)
				if (op != "Add")
					tk.fail("eltwise " + name + ": operation " + op + " (NPUC compiles Add only)");
				// NPUC's Eltwise only shifts: every scale must be 2^-fl, and there is no zero-point term
				EltwiseFL fl{0, 0, 0};
				int *fls[3] = {&fl.fl1, &fl.fl2, &fl.flo};
				const float sc[3] = {s1, s2, so};
				for (int j = 0; j < 3 && !tk.bad; j++) {
					uint32_t u;
					memcpy(&u, &sc[j], 4);
					*fls[j] = 127 - (int)((u >> 23) & 0xff);  // a normal positive float: 2^(exponent - 127) x 1.mantissa
					if ((u & 0x7fffff) != 0 || *fls[j] < 0 || *fls[j] > 31)
						tk.fail("eltwise " + name + ": scale " + std::to_string(sc[j]) +
						        " is not 2^-n with n in 0..31 (NPUC's Eltwise aligns by powers of two only)");
				}
				if (!tk.bad && (z1 != zo || z2 != zo))
					tk.fail("eltwise " + name + ": zero points " + std::to_string(z1) + ", " + std::to_string(z2) + ", " +
					        std::to_string(zo) + " differ (NPUC's Eltwise has no zero-point term)");
				qin(tk, name, ins[0], s1, z1);
				qin(tk, name, ins[1], s2, z2);
				if (tk.bad)
					break;
				bq[name] = {so, zo};
				g_eltwise[name] = fl;
				type = "Eltwise";
				auto ea = std::make_shared<EltwiseAttr>();
				ea->op = op;
				attr = ea;
				snprintf(extra, sizeof extra, "%s, scales in1 %.9g zp %d in2 %.9g zp %d out %.9g zp %d", op.c_str(),
				         (double)s1, z1, (double)s2, z2, (double)so, zo);
			} else if (kw == "concat") {
				// NPUC never reads ConcatAttr on this path (getConcatParam throws for "stensorflow"): it places each
				// input as a channel slice of the output (ChannelSlicedFeatureNode, in input order), so channels only
				int axis = (int)tk.num(1, 1, "axis (NCHW; NPUC concatenates channels only)");
				long n = tk.num(2, 64, "number of inputs");
				for (long j = 0; j < n && !tk.bad; j++)
					ins.push_back(tk.str());
				float so = tk.f32hex("out scale");
				int zo = (int)tk.num(0, 255, "out zp");
				// optional "mixed" (no keyword starts so): the inputs keep their own quantisations
				const bool mixed = tk.more() && tk.t[tk.i] == "mixed";
				if (mixed)
					tk.i++;
				for (auto &in : ins) {
					if (!mixed)
						qin(tk, name, in, so, zo);  // the bytes are placed as they are: every input in the output's quantisation
					else if (ver >= 2 && !bq.count(in))
						tk.fail("concat " + name + ": input " + in + " has no quantisation yet (mixed: a layer's output)");
					if (!layers.empty() && in == layers.front()->name)
						tk.fail("concat " + name + ": input " + in + " is the network input (NPUC would write it straight "
						        "into a channel slice, and its policy maker asserts \"DataNode id isn't found in the NNGraph "
						        "inputs\")");
				}
				if (tk.bad)
					break;
				bq[name] = {so, zo};
				type = "Concat";
				auto ca = std::make_shared<ConcatAttr>();
				ca->axis = axis;
				ca->outScale = {so};
				attr = ca;
				hasConcat = true;
				snprintf(extra, sizeof extra, "axis %d, out scale %.9g zp %d%s", axis, (double)so, zo,
				         mixed ? ", mixed (each input keeps its own quantisation)" : "");
			} else {
				ins.push_back(tk.str());
				type = tk.str();
				if (type != "ReLU" && type != "ReLU6")
					tk.fail("relu " + name + ": type " + type + " (ReLU or ReLU6)");
				if (tk.bad)
					break;
				if (bq.count(ins[0]))
					bq[name] = bq[ins[0]];  // addReLUGraphNode's 1x1 conv keeps the input's quantisation (G)
				attr = std::make_shared<PlainAttr>();
				snprintf(extra, sizeof extra, "%s", type.c_str());
			}
			if (blobs.count(name)) {
				tk.fail("layer " + name + ": duplicate name");
				break;
			}
			std::vector<std::vector<int32_t>> inShapes;
			for (auto &in : ins) {
				auto bi = blobs.find(in);
				if (bi == blobs.end()) {
					tk.fail("layer " + name + ": unknown input " + in);
					break;
				}
				inShapes.push_back({1, bi->second.c, bi->second.h, bi->second.w});
			}
			if (tk.bad)
				break;
			// the output shape, by the layer's own rule (no NPUC code computes these: the HAL's NNConcatLayer has its
			// own computeOutputShape, and there is no HAL Eltwise/ReLU layer)
			os.c = inShapes[0][1], os.h = inShapes[0][2], os.w = inShapes[0][3];
			for (size_t j = 1; j < inShapes.size(); j++) {
				const auto &s = inShapes[j];
				int axis = kw == "concat" ? std::static_pointer_cast<ConcatAttr>(attr)->axis : 0;
				for (int d = 1; d < 4; d++)
					if (d != axis && s[d] != inShapes[0][d])
						tk.fail("layer " + name + ": input " + ins[j] + " is " + std::to_string(s[1]) + "x" +
						        std::to_string(s[2]) + "x" + std::to_string(s[3]) + ", " + ins[0] + " is " +
						        std::to_string(inShapes[0][1]) + "x" + std::to_string(inShapes[0][2]) + "x" +
						        std::to_string(inShapes[0][3]));
				if (axis == 1)
					os.c += s[1];
				else if (axis == 2)
					os.h += s[2];
				else if (axis == 3)
					os.w += s[3];
			}
			if (tk.bad)
				break;
			attrs.push_back(attr);
			auto *nl = static_cast<NNLayer *>(::operator new(sizeof(NNLayer)));
			STEP("NNLayer C2 (" + type + " " + name + ")");
			npuc_NNLayer_C2(nl, name, type, ins, {name}, attr);
			std::shared_ptr<NNLayer> l(nl);
			for (size_t j = 0; j < ins.size(); j++)
				l->shapes[ins[j]] = inShapes[j];
			l->shapes[name] = {1, os.c, os.h, os.w};
			blobs[name] = os;
			std::string inl;
			for (auto &in : ins)
				inl += (inl.empty() ? "" : ", ") + in;
			printf("layer %s %s in %s, %s, out %dx%dx%d\n", name.c_str(), type.c_str(), inl.c_str(), extra, os.c, os.h,
			       os.w);
			print_shapes(l);
			std::string ij;
			for (auto &in : ins)
				ij += (ij.empty() ? "\"" : ", \"") + in + "\"";
			char lb[512];
			snprintf(lb, sizeof lb, "%s{\"name\": \"%s\", \"type\": \"%s\", \"input\": \"%s\", \"inputs\": [%s], "
			         "\"out_shape\": [1, %d, %d, %d], \"pads\": [0, 0, 0, 0]}", first ? "" : ",\n  ", name.c_str(),
			         type.c_str(), ins[0].c_str(), ij.c_str(), os.c, os.h, os.w);
			jl += lb;
			first = false;
			layers.push_back(l);
			last = name;
		} else if (kw == "wscales" || kw == "slopes") {
			std::string name = tk.str();
			auto ci = convByName.find(name);
			if (ci == convByName.end()) {
				tk.fail(kw + " " + name + ": no conv or dwconv of that name before it");
				break;
			}
			ConvAttr &ca = *ci->second;
			long n = tk.num(1, 4096, "n");
			if (!tk.bad && n != ca.numOutput)
				tk.fail(kw + " " + name + ": " + std::to_string(n) + " values for " + std::to_string(ca.numOutput) + " outputs");
			if (kw == "wscales") {
				std::vector<float> ws;
				for (long j = 0; j < n && !tk.bad; j++)
					ws.push_back(tk.f32hex("w scale"));
				if (tk.bad)
					break;
				ca.wScale = ws;
				ca.bScale.clear();
				for (float w : ws)
					ca.bScale.push_back(ca.inScale.at(0) * w);
				printf("wscales %s: %ld per-channel weight scales, first %.9g, last %.9g\n", name.c_str(), n,
				       (double)ws.front(), (double)ws.back());
			} else {
				PerChannel pc;
				for (long j = 0; j < n && !tk.bad; j++)
					pc.slope.push_back((uint32_t)tk.num(1, 65535, "slope"));
				for (long j = 0; j < n && !tk.bad; j++)
					pc.shift.push_back((uint8_t)tk.num(0, 31, "shift"));
				if (tk.bad)
					break;
				g_slopes[name] = pc;
				printf("slopes %s: %ld per-channel slopes and shifts (first %u / %u)\n", name.c_str(), n, pc.slope[0],
				       (unsigned)pc.shift[0]);
			}
		} else {
			tk.fail("unknown keyword '" + kw + "'");
		}
	}
	if (tk.bad || layers.size() < 2) {
		printf("npuc-net: FAIL (input: %s)\n", tk.bad ? tk.why.c_str() : "no layer after the input");
		return 1;
	}
	// the outputs: NPUC's rule (appendNewOutData: a node without successors, or the --intermediate layer), checked
	// against the declared ones
	auto readers = [&](const std::string &b) {
		int n = 0;
		for (auto &m : layers)
			for (auto &in : m->inputs)
				n += in == b && m->name != b;
		return n;
	};
	if (!inter.empty() && (!blobs.count(inter) || inter == layers.front()->name)) {
		printf("npuc-net: FAIL (--intermediate %s: not a layer of the network)\n", inter.c_str());
		return 1;
	}
	for (auto &o : outs) {
		if (!blobs.count(o) || o == layers.front()->name) {
			printf("npuc-net: FAIL (output %s: not a layer of the network)\n", o.c_str());
			return 1;
		}
		if (readers(o) && o != inter) {
			printf("npuc-net: FAIL (output %s is read by %d layer(s): NPUC outputs only the blobs no layer reads "
			       "(NNGraphCgObject::appendNewOutData); the HAL marks nothing else)\n", o.c_str(), readers(o));
			return 1;
		}
	}
	if (!outs.empty())
		for (size_t j = 1; j < layers.size(); j++) {
			const std::string &b = layers[j]->name;
			bool listed = false;
			for (auto &o : outs)
				listed = listed || o == b;
			if (!readers(b) && !listed) {
				printf("npuc-net: FAIL (blob %s is read by no layer, so NPUC makes it a network output, but the output "
				       "line does not list it)\n", b.c_str());
				return 1;
			}
		}
	if (!inter.empty()) {
		bool listed = outs.empty();
		for (auto &o : outs)
			listed = listed || o == inter;
		if (!listed) {
			printf("npuc-net: FAIL (--intermediate %s is not in the output line)\n", inter.c_str());
			return 1;
		}
		// NNGraphCgObject::isIntermediateOutput compares the layer's name with this string (CompileOptions +0xb8,
		// copied to the graph object by Compilation::generateGraph); setCompileOptions leaves it alone
		*reinterpret_cast<std::string *>(static_cast<char *>(opts) + CompileOptionsOff::strB8) = inter;
		printf("intermediate output: %s (CompileOptions +0xb8; read by %d layer(s))\n", inter.c_str(), readers(inter));
	}
	for (auto &o : outs)
		printf("output %s %dx%dx%d (declared)\n", o.c_str(), blobs[o].c, blobs[o].h, blobs[o].w);

	NNNetwork net;
	STEP("append layers");
	for (auto &l : layers)
		if (npuc_append(net, l)) {
			printf("npuc-net: FAIL (append %s: duplicate)\n", l->name.c_str());
			return 2;
		}
	if (ver >= 2 && outs.empty()) {
		// a blob no layer reads, other than the last, would become an extra network output
		for (auto &l : layers) {
			bool used = l->name == last;
			for (auto &m : layers)
				for (auto &in : m->inputs)
					used = used || (in == l->name && m != l);
			if (!used)
				printf("note: blob %s is read by no layer (NPUC makes it an output of the network)\n", l->name.c_str());
		}
	}
	if (hasConcat) {
		// nn-eden-drv's buildNNNetwork ends with NNNetwork::mergeConcatLayer() (concat chains); a no-op without Concat
		STEP("NNNetwork::mergeConcatLayer (NPUC's)");
		if (npuc_mergeConcatLayer)
			npuc_mergeConcatLayer(&net);
		printf("network after mergeConcatLayer: %zu layers\n", net.layers.size());
	}
	NPUCompilerConfig cfg;
	cfg.soc = soc;
	cfg.quantMode = quant;
	cfg.unk8 = 0;
	cfg.inputFormat = "NO_FORMAT";
	NPUCompiler comp{nullptr, &cfg};
	__NCPBuffer buf{nullptr, 0, 0};
	printf("network: %zu layers, output %s, soc %d, quant %d\n", layers.size(), last.c_str(), soc, quant);
	STEP("NPUCompiler::compile (NPUC runs: frontend fillers, GraphIR/LIR, backend NCP)");
	int rc = npuc_compile(&comp, &net, &buf);
	STEP("after compile");
	for (auto &e : g_slopes)
		if (e.second.calls == 0) {
			printf("npuc-net: FAIL (slopes %s: NPUC's setScale never ran for that layer)\n", e.first.c_str());
			return 2;
		}
	printf("NPUCompiler::compile = %d, addr %p, size %d\n", rc, buf.addr, buf.size);
	if (rc != 0 || !buf.addr || buf.size < 0x80) {
		printf("npuc-net: FAIL (compile)\n");
		return 2;
	}
	FILE *f = fopen(out, "wb");
	if (!f || fwrite(buf.addr, 1, (size_t)buf.size, f) != (size_t)buf.size || fclose(f) != 0) {
		printf("npuc-net: FAIL (cannot write %s)\n", out);
		return 2;
	}
	uint32_t w[32];
	memcpy(w, buf.addr, sizeof w);
	printf("NCP %s: %d bytes, magic1 0x%08" PRIx32 " version %" PRIu32 " magic2 0x%08" PRIx32 "\n", out, buf.size,
	       w[0], w[1], w[31]);
	if (w[0] != 0x0C0FFEE0u || w[1] != 22 || w[31] != 0xC0DEC0DEu) {
		printf("npuc-net: FAIL (not an NCP v22 header)\n");
		return 3;
	}
	std::string pj = std::string(out);
	pj = (pj.size() > 4 && pj.compare(pj.size() - 4, 4, ".ncp") == 0 ? pj.substr(0, pj.size() - 4) : pj) + ".json";
	FILE *j = fopen(pj.c_str(), "w");
	if (!j) {
		printf("npuc-net: FAIL (cannot write %s)\n", pj.c_str());
		return 2;
	}
	Shape is = blobs[layers.front()->name], os = blobs[last];
	fprintf(j, "{\"format\": \"npuc-net-out/1\", \"ncp\": \"%s\", \"ncp_size\": %d, \"soc\": %d, \"quant_mode\": %d,\n"
	           " \"input\": \"%s\", \"in_shape\": [1, %d, %d, %d], \"output\": \"%s\", \"out_shape\": [1, %d, %d, %d],\n"
	           " \"layers\": [", out, buf.size, soc, quant, layers.front()->name.c_str(), is.c, is.h, is.w, last.c_str(),
	        os.c, os.h, os.w);
	fputs(jl.c_str(), j);
	fprintf(j, "]");
	if (!outs.empty()) {
		// the declared outputs, in the file's order (NPUC's own order is its node order: its log's "Network Output:")
		fprintf(j, ",\n \"outputs\": [");
		for (size_t k = 0; k < outs.size(); k++) {
			const Shape &b = blobs[outs[k]];
			fprintf(j, "%s{\"name\": \"%s\", \"out_shape\": [1, %d, %d, %d]}", k ? ", " : "", outs[k].c_str(), b.c,
			        b.h, b.w);
		}
		fprintf(j, "]");
	}
	if (!inter.empty())
		fprintf(j, ",\n \"intermediate\": \"%s\"", inter.c_str());
	fprintf(j, "}\n");
	if (fclose(j) != 0) {
		printf("npuc-net: FAIL (cannot write %s)\n", pj.c_str());
		return 2;
	}
	printf("FILE %s\nFILE %s\n", out, pj.c_str());
	printf("npuc-net: PASS\n");
	return 0;
}
