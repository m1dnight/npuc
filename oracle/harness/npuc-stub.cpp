// npuc-stub: a stand-in for Samsung's NPUC, for the host test of npuc-net only (npu/host-test.sh). Linked with
// npuc-net.cpp into a static x86_64 Android executable that runs on the build host: it defines the NPUC entry points
// npuc-net calls (by the same symbol names), computes output shapes with TFLite's SAME/VALID rule (= NPUC's, V:
// NNConvLayer::computeOutputShape) and, instead of compiling, prints every layer as npuc-net built it, so the test
// can check each value against npu/npunet.py's lowered.json. Never runs on the phone; contains no vendor code.
#include "npuc.h"

#include <cstdio>
#include <cstring>
#include <new>

using namespace NPUC;

NNLayer::~NNLayer() {}
void NNLayer::computeOutputShape(const std::vector<std::vector<int32_t>> &) {}
void ConvAttr::printAttr() {}
ConvAttr::~ConvAttr() {}
void PoolAttr::printAttr() {}
PoolAttr::~PoolAttr() {}

static void same_valid(const std::string &padType, int in, int k, int s, int &out, int &p0, int &p1)
{
	if (padType == "SAME") {
		out = (in + s - 1) / s;
		int tot = (out - 1) * s + k - in;
		p0 = tot / 2;
		p1 = tot - p0;
	} else {
		out = (in - k) / s + 1;
		p0 = p1 = 0;
	}
}

void NNConvLayer::computeOutputShape(const std::vector<std::vector<int32_t>> &in)
{
	auto a = std::static_pointer_cast<ConvAttr>(attr);
	int ho, wo, t, b, l, r;
	same_valid(a->padType, in[0][2], a->kernel[0], a->stride[2], ho, t, b);
	same_valid(a->padType, in[0][3], a->kernel[1], a->stride[3], wo, l, r);
	a->pad = {t, b, l, r};
	shapes[outputs[0]] = {1, a->numOutput, ho, wo};
}

void NNPoolLayer::computeOutputShape(const std::vector<std::vector<int32_t>> &in)
{
	auto a = std::static_pointer_cast<PoolAttr>(attr);
	int ho, wo, t, b, l, r;
	same_valid(a->padType, in[0][2], a->kernel[2], a->stride[2], ho, t, b);
	same_valid(a->padType, in[0][3], a->kernel[3], a->stride[3], wo, l, r);
	a->pad = {t, b, l, r};
	shapes[outputs[0]] = {1, in[0][1], ho, wo};
}

static void init(NNLayer *s, std::string &n, std::string &t, std::vector<std::string> &i, std::vector<std::string> &o)
{
	s->name = n;
	s->type = t;
	s->inputs = i;
	s->outputs = o;
}

void stub_NNLayer_C2(NNLayer *self, std::string n, std::string t, std::vector<std::string> i, std::vector<std::string> o,
                     std::shared_ptr<LayerAttr> a)
	__asm__("_ZN4NPUC7NNLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_"
	        "NS5_IS7_EEEESA_NS1_10shared_ptrINS_9LayerAttrEEE");
void stub_NNLayer_C2(NNLayer *self, std::string n, std::string t, std::vector<std::string> i, std::vector<std::string> o,
                     std::shared_ptr<LayerAttr> a)
{
	new (self) NNLayer();
	init(self, n, t, i, o);
	self->attr = a;
}

void stub_NNConvLayer_C2(NNConvLayer *self, std::string n, std::string t, std::vector<std::string> i,
                         std::vector<std::string> o, std::shared_ptr<ConvAttr> a)
	__asm__("_ZN4NPUC11NNConvLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vector"
	        "IS7_NS5_IS7_EEEESA_NS1_10shared_ptrINS_8ConvAttrEEE");
void stub_NNConvLayer_C2(NNConvLayer *self, std::string n, std::string t, std::vector<std::string> i,
                         std::vector<std::string> o, std::shared_ptr<ConvAttr> a)
{
	new (self) NNConvLayer();
	init(self, n, t, i, o);
	self->attr = a;
}

void stub_NNPoolLayer_C2(NNPoolLayer *self, std::string n, std::string t, std::vector<std::string> i,
                         std::vector<std::string> o, std::shared_ptr<PoolAttr> a)
	__asm__("_ZN4NPUC11NNPoolLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vector"
	        "IS7_NS5_IS7_EEEESA_NS1_10shared_ptrINS_8PoolAttrEEE");
void stub_NNPoolLayer_C2(NNPoolLayer *self, std::string n, std::string t, std::vector<std::string> i,
                         std::vector<std::string> o, std::shared_ptr<PoolAttr> a)
{
	new (self) NNPoolLayer();
	init(self, n, t, i, o);
	self->attr = a;
}

static char options[0x168];
void *stub_GetInstance() __asm__("_ZN4NPUC14CompileOptions11GetInstanceEv");
void *stub_GetInstance() { return options; }

static uint32_t bits(float f)
{
	uint32_t u;
	memcpy(&u, &f, 4);
	return u;
}

int stub_compile(NPUCompiler *, NNNetwork *, __NCPBuffer *) __asm__("_ZN4NPUC11NPUCompiler7compileEPNS_9NNNetworkEPNS_11__NCPBufferE");
int stub_compile(NPUCompiler *c, NNNetwork *net, __NCPBuffer *out)
{
	printf("stub: compile soc %d quant %d format %s, %zu layers\n", c->config->soc, c->config->quantMode,
	       c->config->inputFormat.c_str(), net->layers.size());
	for (auto &l : net->layers) {
		printf("stub: layer %s type %s in %s out %s", l->name.c_str(), l->type.c_str(), l->inputs[0].c_str(),
		       l->outputs[0].c_str());
		if (l->type == "Convolution") {
			auto a = std::static_pointer_cast<ConvAttr>(l->attr);
			uint64_t sum = 0;
			for (size_t k = 0; k < a->kernelData->size(); k++)
				sum = sum * 31 + (*a->kernelData)[k];
			printf(" conv %s cin %d cout %d k %dx%d stride %dx%d act %d in 0x%08x/%d w 0x%08x/%d out 0x%08x/%d "
			       "weights %zu hash %llu bias", a->padType.c_str(), a->numInput, a->numOutput, a->kernel[0],
			       a->kernel[1], a->stride[2], a->stride[3], a->fusedActivation, bits(a->inScale[0]), a->inZp[0],
			       bits(a->wScale[0]), a->wZp[0], bits(a->outScale[0]), a->outZp[0], a->kernelData->size(),
			       (unsigned long long)sum);
			for (int b : *a->biasData)
				printf(" %d", b);
		} else if (l->type == "Pooling") {
			auto a = std::static_pointer_cast<PoolAttr>(l->attr);
			printf(" pool %s %s k %dx%d stride %dx%d act %d in 0x%08x/%d out 0x%08x/%d", a->poolType.c_str(),
			       a->padType.c_str(), a->kernel[2], a->kernel[3], a->stride[2], a->stride[3], a->fusedActivation,
			       bits(a->inScale[0]), a->inZp[0], bits(a->outScale[0]), a->outZp[0]);
		}
		printf("\n");
	}
	static uint32_t ncp[64];
	ncp[0] = 0x0C0FFEE0u;
	ncp[1] = 22;
	ncp[31] = 0xC0DEC0DEu;
	out->addr = ncp;
	out->size = sizeof ncp;
	return 0;
}
