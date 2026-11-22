/*
 * npuc-probe: PORT-32 N3a step 2 (dossier/npu-compiler.md section 6).
 *
 * Loads Samsung's on-device NPU compiler (NPUC, libnpuc_*.so from the phone's own stock vendor partition) into a
 * plain Bionic process, outside Android: no binder, no HAL, no property service, no linker namespace config.
 * Our own code; it contains no vendor byte. The vendor libraries and the stock Bionic runtime are supplied at run
 * time from a local directory (never shipped with this source).
 *
 * The same aarch64 binary runs
 *   - on the phone:  /root/npuc/linker64 /root/npuc/npuc-probe   (LD_LIBRARY_PATH=/root/npuc)
 *   - under qemu:    qemu-aarch64 <root>/linker64 <root>/npuc-probe (LD_LIBRARY_PATH=<root>)
 * Bionic's linker is run as a program (Android Q+), so the binary's PT_INTERP (/system/bin/linker64) is not used.
 *
 * What it checks, in order (exit code in brackets on failure):
 *   1. dlopen("libnpuc_api.so", RTLD_NOW) loads all eight libnpuc_*.so through DT_NEEDED with every symbol
 *      bound, then each is confirmed with RTLD_NOLOAD                                                    [2]
 *      Only the root is dlopen()ed: libnpuc_graph and libnpuc_backend are underlinked (they use symbols of
 *      libnpuc_common / libnpuc_cmdq without listing them in DT_NEEDED) and bind only inside libnpuc_api's
 *      DT_NEEDED closure, as in nn-eden-drv (whose NEEDED list has them all). Loading them one by one fails
 *      even after RTLD_GLOBAL loads of the providers (phone run 2026-11-21 104734Z).
 *   2. the entry points the NNAPI HAL (nn-eden-drv) uses resolve, in the library and at the offset
 *      llvm-nm showed on the host (the right build of the right library)                               [3]
 *   3. with --call-options: NPUC::CompileOptions::GetInstance() returns, and the first --dump bytes (default
 *      0x168, the whole object) of the singleton are printed in hex (its defaults: npuc.h's oracle)    [4]
 * PASS line: "npuc-probe: PASS".
 *
 * NPUC imports one unversioned symbol no NEEDED library provides: __android_log_print (the HAL gets it from
 * liblog). We define it here and export it (-Wl,--export-dynamic), so liblog is optional; --liblog dlopen()s the
 * stock liblog.so RTLD_GLOBAL first instead (ours still wins: the executable comes first in the global group).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

/* ---- liblog replacement ------------------------------------------------------------------------------------ */

static int alog_quiet;

__attribute__((visibility("default"))) int __android_log_print(int prio, const char *tag, const char *fmt, ...)
{
	static const char lv[] = "??VDIWEFS";
	va_list ap;
	if (alog_quiet)
		return 0;
	fprintf(stderr, "alog %c/%s: ", (prio >= 0 && prio < 9) ? lv[prio] : '?', tag ? tag : "-");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	if (!*fmt || fmt[strlen(fmt) - 1] != '\n')
		fputc('\n', stderr);
	return 1;
}

__attribute__((visibility("default"))) int __android_log_write(int prio, const char *tag, const char *msg)
{
	return __android_log_print(prio, tag, "%s", msg ? msg : "");
}

__attribute__((visibility("default"))) int __android_log_vprint(int prio, const char *tag, const char *fmt, va_list ap)
{
	char buf[1024];
	vsnprintf(buf, sizeof buf, fmt, ap);
	return __android_log_print(prio, tag, "%s", buf);
}

/* ---- what we expect -------------------------------------------------------------------------------------- */

/* The root, then the libraries its DT_NEEDED closure must have brought in (G980FXXSNHYB1 vendor build). */
static const char *const root_lib = "libnpuc_api.so";
static const char *const libs[] = {
	"libnpuc_common.so", "libnpuc_graph.so", "libnpuc_cmdq.so", "libnpuc_backend.so", "libnpuc_template.so",
	"libnpuc_frontend.so", "libnpuc_controller.so", "libnpuc_api.so",
};

struct want {
	const char *lib;       /* where llvm-nm -D --defined-only found it */
	unsigned long off;     /* its st_value there (0 = do not check) */
	const char *sym;
	const char *what;
};

/* Offsets: G980FXXSNHYB1 vendor libs (output/npu/vendor-libs/SHA256SUMS). A mismatch is reported, not fatal:
 * another build of NPUC would move them. */
static const struct want wants[] = {
	{ "libnpuc_api.so", 0x8a8, "_ZN4NPUC11NPUCompiler7compileEPNS_9NNNetworkEPNS_11__NCPBufferE",
	  "NPUC::NPUCompiler::compile(NNNetwork*, __NCPBuffer*)" },
	{ "libnpuc_api.so", 0x8fc, "_ZN4NPUC11NPUCompiler20isSupportedOperationENS_6OpTypeERNS_25SupportedOperationOptionsE",
	  "NPUC::NPUCompiler::isSupportedOperation(OpType, SupportedOperationOptions&)" },
	{ "libnpuc_common.so", 0x4973c, "_ZN4NPUC14CompileOptions11GetInstanceEv",
	  "NPUC::CompileOptions::GetInstance()" },
	{ "libnpuc_common.so", 0x49628, "_ZN4NPUC14CompileOptionsC1Ev", "NPUC::CompileOptions::CompileOptions()" },
	{ "libnpuc_controller.so", 0x3448, "_ZN4NPUC14NPUCController3runEPNS_9NNNetworkEPNS_11__NCPBufferE",
	  "NPUC::NPUCController::run(NNNetwork*, __NCPBuffer*)" },
	{ "libnpuc_controller.so", 0x3a5c, "_ZN4NPUC14NPUCController17setCompileOptionsEv",
	  "NPUC::NPUCController::setCompileOptions()" },
	{ "libnpuc_frontend.so", 0xb3f80, "_ZN4NPUC9NNNetworkD1Ev", "NPUC::NNNetwork::~NNNetwork()" },
	{ "libnpuc_frontend.so", 0xb42b0, "_ZN4NPUC9NNNetwork12setUpNetworkEv", "NPUC::NNNetwork::setUpNetwork()" },
};

/* ---- helpers --------------------------------------------------------------------------------------------- */

static int phdr_cb(struct dl_phdr_info *i, size_t sz, void *arg)
{
	(void)sz;
	(void)arg;
	if (i->dlpi_name && strstr(i->dlpi_name, ".so"))
		printf("  map 0x%012lx %s\n", (unsigned long)i->dlpi_addr, i->dlpi_name);
	return 0;
}

static const char *base_name(const char *p)
{
	const char *s = p ? strrchr(p, '/') : NULL;
	return s ? s + 1 : (p ? p : "?");
}

static void hexdump(const void *p, size_t n)
{
	const unsigned char *b = p;
	for (size_t o = 0; o < n; o += 16) {
		printf("  +0x%03zx:", o);
		for (size_t k = 0; k < 16 && o + k < n; k++)
			printf(" %02x", b[o + k]);
		printf("\n");
	}
}

static void usage(void)
{
	fprintf(stderr, "usage: npuc-probe [--liblog] [--quiet-log] [--call-options] [--dump <bytes>]\n");
}

int main(int argc, char **argv)
{
	int use_liblog = 0, call_opts = 0, rc = 0;
	size_t dump = 0x168;  /* sizeof(CompileOptions): the static object ends at +0x168 (npuc.h) */
	struct utsname u;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--liblog"))
			use_liblog = 1;
		else if (!strcmp(argv[i], "--quiet-log"))
			alog_quiet = 1;
		else if (!strcmp(argv[i], "--call-options"))
			call_opts = 1;
		else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
			dump = strtoul(argv[++i], NULL, 0);
		else {
			usage();
			return 1;
		}
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (uname(&u) == 0)
		printf("npuc-probe: %s %s %s, pid %d\n", u.sysname, u.release, u.machine, (int)getpid());
	printf("npuc-probe: LD_LIBRARY_PATH=%s\n", getenv("LD_LIBRARY_PATH") ? getenv("LD_LIBRARY_PATH") : "(unset)");

	if (use_liblog) {
		void *h = dlopen("liblog.so", RTLD_NOW | RTLD_GLOBAL);
		printf("load liblog.so: %s\n", h ? "OK" : dlerror());
	}

	/* 1. load the root; its DT_NEEDED closure brings the other seven */
	if (!dlopen(root_lib, RTLD_NOW | RTLD_GLOBAL)) {
		printf("load %s: FAIL %s\n", root_lib, dlerror());
		printf("npuc-probe: FAIL (load)\n");
		return 2;
	}
	printf("load %s: OK\n", root_lib);
	for (size_t i = 0; i < sizeof libs / sizeof libs[0]; i++) {
		void *h = dlopen(libs[i], RTLD_NOW | RTLD_NOLOAD);
		if (!h) {
			printf("loaded %s: FAIL not in the process (%s)\n", libs[i], dlerror());
			rc = 2;
		} else
			printf("loaded %s: OK\n", libs[i]);
	}
	if (rc) {
		printf("npuc-probe: FAIL (load)\n");
		return rc;
	}
	dl_iterate_phdr(phdr_cb, NULL);

	/* 2. resolve */
	void *getinst = NULL;
	for (size_t i = 0; i < sizeof wants / sizeof wants[0]; i++) {
		const struct want *w = &wants[i];
		void *a = dlsym(RTLD_DEFAULT, w->sym);
		Dl_info di;
		if (!a) {
			printf("sym %s: FAIL not found (%s)\n", w->what, dlerror());
			rc = 3;
			continue;
		}
		memset(&di, 0, sizeof di);
		dladdr(a, &di);
		unsigned long off = (unsigned long)((uintptr_t)a - (uintptr_t)di.dli_fbase);
		const char *in = base_name(di.dli_fname);
		int same = !strcmp(in, w->lib) && (!w->off || off == w->off);
		printf("sym %s: OK %s+0x%lx%s\n", w->what, in, off,
		       same ? "" : "  (differs from the G980FXXSNHYB1 build: not that library or not that offset)");
		if (!strcmp(w->sym, "_ZN4NPUC14CompileOptions11GetInstanceEv"))
			getinst = a;
	}
	{
		void *a = dlsym(RTLD_DEFAULT, "__android_log_print");
		Dl_info di;
		memset(&di, 0, sizeof di);
		if (a)
			dladdr(a, &di);
		printf("sym __android_log_print: %s\n", a ? base_name(di.dli_fname) : "not found");
	}
	if (rc) {
		printf("npuc-probe: FAIL (symbols)\n");
		return rc;
	}

	/* 3. the options singleton */
	if (call_opts) {
		void *(*gi)(void) = (void *(*)(void))getinst;
		void *o = gi();
		if (!o) {
			printf("CompileOptions::GetInstance(): NULL\n");
			printf("npuc-probe: FAIL (options)\n");
			return 4;
		}
		Dl_info di;
		memset(&di, 0, sizeof di);
		void *vt = *(void **)o;
		dladdr(vt, &di);
		printf("CompileOptions::GetInstance() = %p, vptr %p (%s+0x%lx)\n", o, vt, base_name(di.dli_fname),
		       (unsigned long)((uintptr_t)vt - (uintptr_t)di.dli_fbase));
		hexdump(o, dump);
	}
	printf("npuc-probe: PASS\n");
	return 0;
}
