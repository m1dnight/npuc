// npuc.h: the object model of Samsung's on-device NPU compiler (NPUC 1.36.1.a, "v1.4.1" in its logs), as far as
// PORT-32 N3a could reconstruct it. Our own header: written from offsets, sizes and call sequences read in the
// G980FXXSNHYB1 binaries (nn-eden-drv's NNNetworkBuilder, libnpuc_frontend/controller/common/api/backend), with
// Ghidra 12.2 headless and llvm-objdump. No vendor header, table or byte is copied here.
//
// Evidence tags on every member: V = read in the code (offset, type and meaning seen), O = offset and type seen,
// meaning guessed, G = guess. Addresses are the file's own (Ghidra shows them +0x100000).
//
// ABI: libc++ ABI v1 (std::__ndk1 for NPUC, std::__1 for nn-eden-drv: same layouts), aarch64 Itanium C++ ABI.
// NPUC links its libc++ statically and exports it from libnpuc_common. A harness built with the NDK (r27d) must
// keep its own libc++ private (-static-libstdc++, -fvisibility=hidden, -Wl,--exclude-libs,ALL) and must not
// interpose NPUC's; npuc-layout-check.cpp asserts the sizes below against the NDK's headers.
//
// How Android drives it (nn-eden-drv, CompilerManager::compile @0x692b4 and NNNetworkBuilder, V):
//   1. NNNetwork net (make_shared, 0x60 bytes); one NNLayer per op, appended with NNNetwork::appendLayers
//   2. NPUC::CompileOptions::GetInstance() (a static singleton in libnpuc_common); the builder clears +0x13c
//   3. NPUCompiler::compile(&net, &buf) -> NPUCController{compiler->config}.run(&net, &buf):
//      setCompileOptions() copies the config into CompileOptions, Compilation::run (frontend -> GraphIR, LIR),
//      BeCompilation::run (backend -> NCP into buf); 0 = success ("NCP binary is generated successfully.")
#pragma once
// The NPUC classes keep default visibility so that a harness built with -fvisibility=hidden still binds their
// vtables and constructors to libnpuc_frontend's exported symbols.
#define NPUC_API __attribute__((visibility("default")))
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace NPUC {

// ---- __NCPBuffer: the compiler's output (V: CompilerManager::compile allocates 0x10 bytes, zeroed, and logs
// "Compile is complete, addr=%p, size=%d" with +0 and +8; then hands addr/size to the Eden custom op) ----------
struct __NCPBuffer {
	void *addr;      // V  the NCP (header v22 + body), allocated by NPUC (G: malloc/new; never freed by us)
	int32_t size;    // V  its length in bytes
	int32_t _pad;
};
static_assert(sizeof(__NCPBuffer) == 0x10, "__NCPBuffer");

// ---- SOCType (V: HAL::DeviceFactory::createDevice @libnpuc_common 0x4c424, SupportedOperationFactory,
// IsaProcCreator::CreateIsa @libnpuc_backend 0xed354: 0 -> Makalu ISA, 1..3 -> NPU2020 ISA) --------------------
enum SOCType : int32_t {
	SOC_MAKALU = 0,   // V  NPUDeviceMakalu, MakaluIsaProcessor, NCPObjectGeneratorMakalu (hdr v9): the 9820 (G)
	SOC_NPU2020 = 1,  // V  NPUDevice2020 + NPU2020 ISA
	SOC_NEUS = 2,     // V  NPUDeviceNeus + NPU2020 ISA. G: the 990: the stock 990 models say "SOC_TYPE neus"
	SOC_KITT = 3,     // V  NPUDeviceKitt + NPU2020 ISA (G: another 2020-generation SoC)
};

// ---- The config an NPUCompiler points to (V: NPUCController::setCompileOptions @libnpuc_controller 0x3a5c reads
// it through NPUCController's only member). Who fills it in nn-eden-drv was not found (G: the HAL's init). ------
struct NPUCompilerConfig {
	int32_t soc;               // V  SOCType -> CompileOptions.soc (+0x130); 0 -> target "G2", else "SRR"
	int32_t quantMode;         // V  3 = "quasi-symmetric", 1 or other = "asymmetric" (the default)
	int32_t unk8;              // O  copied to CompileOptions +0x140
	int32_t _pad;
	std::string inputFormat;   // V  "Input format is %s" -> CompileOptions.inputFormat (+0xd0); default "NO_FORMAT"
};
static_assert(offsetof(NPUCompilerConfig, inputFormat) == 0x10, "NPUCompilerConfig");

// ---- NPUCompiler (V: compile @libnpuc_api 0x8a8 is `NPUCController c{this->+8}; return c.run(net, buf);`;
// no constructor is exported: the object is plain data) ------------------------------------------------------
struct NNNetwork;
struct NPUCompiler {
	void *unk0;                   // G  not read by compile()
	NPUCompilerConfig *config;    // V
};
// Exported: int NPUC::NPUCompiler::compile(NNNetwork*, __NCPBuffer*)  (this in x0; 0 = OK, the HAL maps != 0 to
// "Fail to compile NNNetwork to get NCP.")
using compile_fn = int (*)(NPUCompiler *self, NNNetwork *net, __NCPBuffer *out);
constexpr const char *kSymCompile = "_ZN4NPUC11NPUCompiler7compileEPNS_9NNNetworkEPNS_11__NCPBufferE";

// ---- CompileOptions: a singleton, `static CompileOptions inst` in GetInstance @libnpuc_common 0x4973c
// (.bss 0xf80c8, 0x168 bytes, guard 0xf80c0). Values: GetInstance's defaults / what setCompileOptions writes. --
// Accessed as raw offsets (the object belongs to NPUC; we never construct or destroy one).
namespace CompileOptionsOff {
constexpr size_t vptr = 0x000;         // V  _ZTVN4NPUC14CompileOptionsE (virtual ~CompileOptions only)
constexpr size_t target = 0x010;       // V  std::string: "" -> setCompileOptions: "G2" (soc 0) / "SRR"
constexpr size_t framework = 0x028;    // V  std::string: "" -> setCompileOptions: "stensorflow" (11 chars, as is)
constexpr size_t str40 = 0x040;        // O  std::string, default "all"
constexpr size_t str58 = 0x058;        // O  std::string, default ""
constexpr size_t str70 = 0x070;        // O  std::string, default "TEST"
constexpr size_t str88 = 0x088;        // O  std::string
constexpr size_t strA0 = 0x0a0;        // O  std::string
constexpr size_t strB8 = 0x0b8;        // O  std::string
constexpr size_t inputFormat = 0x0d0;  // V  std::string, default "NO_FORMAT", from the config
constexpr size_t quantEnumEC = 0x0ec;  // V  int: 2 for asymmetric, 0 for quasi-symmetric (G: an internal enum)
constexpr size_t flagF0 = 0x0f0;       // O  bool: 1 by default, setCompileOptions clears it
constexpr size_t intF4 = 0x0f4;        // O  int: 16 by default, setCompileOptions sets 8 (G: a bit width)
constexpr size_t str100 = 0x100;       // O  std::string
constexpr size_t str118 = 0x118;       // O  std::string
constexpr size_t soc = 0x130;          // V  int: SOCType, from the config
constexpr size_t int138 = 0x138;       // O  int: setCompileOptions writes 1
constexpr size_t int13c = 0x13c;       // O  int: setCompileOptions 0, NNNetworkBuilder::buildNNNetwork 0
constexpr size_t int140 = 0x140;       // O  int: config.unk8
constexpr size_t quantMode = 0x144;    // V  int: 1 asymmetric / 3 quasi-symmetric
constexpr size_t map150 = 0x150;       // O  std::map<...> (destroyed by ~CompileOptions)
constexpr size_t size = 0x168;         // V  (last member ends at 0x168)
}  // namespace CompileOptionsOff
constexpr const char *kSymGetInstance = "_ZN4NPUC14CompileOptions11GetInstanceEv";
// Note: setCompileOptions() runs at the start of every compile and overwrites the fields it owns (target,
// framework, flagF0, intF4, soc, int138, quantEnumEC, quantMode, int140, inputFormat): set the config, not these.

// ---- LayerAttr and the attributes ---------------------------------------------------------------------------
// LayerAttr (V: typeinfo _ZTIN4NPUC9LayerAttrE exported, vtable not): just the vptr. Vtable order of every
// attribute (V, ConvAttr's relocations): [printAttr, ~D2, ~D0].
class NPUC_API LayerAttr {
public:
	virtual void printAttr() = 0;
	virtual ~LayerAttr() {}
};
static_assert(sizeof(LayerAttr) == 8, "LayerAttr");

// InputAttr (V: nn-eden-drv's own class, not in NPUC; shared_ptr_emplace of 0x30 = 0x18 + 0x18): N,C,H,W.
// NPUC never sees its type (no typeinfo in NPUC, G: only the layer's shape map is read), so ours is our own.
class NPUC_API InputAttr : public LayerAttr {
public:
	int32_t shape[4] = {0, 0, 0, 0};  // V  copied from the input shape vector [0..3] (NCHW, G for the order)
	void printAttr() override {}
};
static_assert(sizeof(InputAttr) == 0x18, "InputAttr");

// ConvAttr: 0x1f8 bytes (V: make_shared in loadConvAttrForConvolutionLayer @0x71b60 news 0x210 = 0x18 + 0x1f8),
// vtable and typeinfo from libnpuc_frontend (_ZTVN4NPUC8ConvAttrE, _ZTIN4NPUC8ConvAttrE, exported). printAttr and
// the destructor are declared here and defined only in NPUC, so the compiler takes NPUC's vtable (the key
// function is printAttr). Allocate with `new` and own with shared_ptr<ConvAttr>(p): deletion then goes through the
// vtable's D0 (no D1 is exported).
class NPUC_API ConvAttr : public LayerAttr {
public:
	ConvAttr() {}
	void printAttr() override;        // libnpuc_frontend, 0x93b50
	~ConvAttr() override;             // libnpuc_frontend, D2 0x96798 / D0 0x96a38 (never called directly)

	std::vector<int32_t> pad;         // +0x008 V  {top, bottom, left, right} (NNAPI explicit pads 2,3,0,1)
	std::vector<int32_t> stride;      // +0x020 V  {1, 1, stride_h, stride_w}
	std::vector<int32_t> dilation;    // +0x038 V  {1, 1, dilation_h, dilation_w}
	int32_t group = 0;                // +0x050 V  1 for "Conv2D"; = numInput for a depthwise conv (setKernelInfo);
	                                  //   read by NNGraphAttribute::getConvolutionParam (-> param +0x2c), isDwConvolution
	                                  //   and DwOptSupported (docs/format/depthwise.md)
	int32_t numInput = 0;             // +0x054 V  input channels (setKernelInfo arg 1 = input shape [1])
	int32_t numOutput = 0;            // +0x058 V  output channels for "Conv2D" (setKernelInfo arg 2 = filter dim 0)
	int32_t _pad5c = 0;               // +0x05c
	std::vector<int32_t> kernel;      // +0x060 V  {kernel_h, kernel_w}
	std::string attrType;             // +0x078 V  "Conv2D" (also "Conv2DBackpropInput"; others = depthwise)
	std::string padType;              // +0x090 V  "EXPLICIT" (10-input NNAPI form), "SAME" or "VALID"
	std::string kernelName;           // +0x0a8 V  "<operand name>_<operand index>"
	std::string biasName;             // +0x0c0 V  idem for the bias operand
	bool hasBias = false;             // +0x0d8 V  set to 1 by the builder
	// The five data slots. Correction of 2026-11-21 (phone run 105104Z, SIGSEGV at address 0 in
	// NNGraphNodeBiasONFiller::setBias): the HAL's control-block indices 0x23/0x27 are object offsets 0x100/0x120
	// (index - 3) * 8, not 0xf0/0x110. Read back on NPUC's side (V): KernelONFiller::setKernel (frontend 0x9cbb4)
	// copies +0x100 into saveKernelData(shared_ptr<BinMemoryView<int8>>&, const shared_ptr<vector<uint8_t>>&)
	// ("WithSubstracting128" when quantMode is 3); BiasONFiller::setBias (0x9c6b0) dereferences +0x120 without a
	// null check and reads it as int32 elements.
	std::shared_ptr<void> data0;      // +0x0e0 G  not set on the quantized path (G: float kernel)
	std::shared_ptr<void> data1;      // +0x0f0 G  not set on the quantized path
	std::shared_ptr<std::vector<uint8_t>> kernelData;  // +0x100 V  the filter bytes (TENSOR_QUANT8_ASYMM, uint8),
	                                  //   count = numOutput*numInput*kh*kw (G: OIHW after the HAL's NCHW conversion;
	                                  //   a saveKernelDataWithConvertFromNHWCtoNCHW variant exists)
	std::shared_ptr<void> data3;      // +0x110 G  not set on the quantized path (G: float bias)
	std::shared_ptr<std::vector<int32_t>> biasData;   // +0x120 V  the int32 bias (TENSOR_INT32), numOutput values
	std::vector<float> inScale;       // +0x130 V  "Input scale are"
	std::vector<int32_t> inZp;        // +0x148 V  "Input zp are"
	std::vector<float> outScale;      // +0x160 V  "Output scale are"
	std::vector<int32_t> outZp;       // +0x178 V  "Output zp are"
	std::vector<float> wScale;        // +0x190 V  "Weight scale are" (one per tensor or per channel)
	std::vector<int32_t> wZp;         // +0x1a8 V  "Weight zp are"
	std::vector<float> bScale;        // +0x1c0 V  "Bias scale are"
	std::vector<int32_t> bZp;         // +0x1d8 V  "Bias zp are"
	int32_t fusedActivation = 0;      // +0x1f0 V  0 none, 1 relu (setFusedActivation; 2..5 handled apart, G)
	int32_t _pad1f4 = 0;
};
constexpr const char *kSymConvAttrVtable = "_ZTVN4NPUC8ConvAttrE";

// PoolAttr: 0xe8 bytes (V: nn-eden-drv NNNetworkBuilder::loadPoolAttrForPoolLayer @0x173c94 make_shared news 0x100 =
// 0x18 + 0xe8 and stores every member below; NPUC's ~PoolAttr D2 @libnpuc_frontend 0x8ff7c frees the same members;
// printAttr's labels "PoolType is", "PadType is", "Stride are", "Kernel are", in/out scale and zp, "FusedActivation is").
// Vtable, typeinfo, printAttr and D2/D0 are exported by libnpuc_frontend (_ZTVN4NPUC8PoolAttrE): the key function is
// printAttr, declared and never defined here, as for ConvAttr. Read back by NPUC's NNPoolLayer::computeOutputShape
// (@frontend 0x90f98): stride +0x20 and kernel +0x38 as 4-element NCHW vectors (indices 2, 3), pads written to +0x08;
// it knows "SAME", "VALID" and "EXPLICIT" (symmetric pads only: "Padding in Pooling is Not Symmetrical!").
class NPUC_API PoolAttr : public LayerAttr {
public:
	PoolAttr() {}
	void printAttr() override;        // libnpuc_frontend, 0x8eafc
	~PoolAttr() override;             // libnpuc_frontend, D2 0x8ff7c / D0 0x90054 (never called directly)

	std::vector<int32_t> pad;         // +0x08 V  {top, bottom, left, right} (NNAPI config words 2, 3, 0, 1)
	std::vector<int32_t> stride;      // +0x20 V  {1, 1, stride_h, stride_w}
	std::vector<int32_t> kernel;      // +0x38 V  {1, 1, filter_h, filter_w} (4 elements: not ConvAttr's {kh, kw})
	std::string padType;              // +0x50 V  "EXPLICIT" (10-input NNAPI form), "SAME" or "VALID"
	std::string poolType;             // +0x68 V  "AvgPool" (AVERAGE_POOL_2D) or "MaxPool" (MAX_POOL_2D)
	std::vector<float> inScale;       // +0x80 V  the input operand's scale(s)
	std::vector<int32_t> inZp;        // +0x98 V
	std::vector<float> outScale;      // +0xb0 V
	std::vector<int32_t> outZp;       // +0xc8 V
	int32_t fusedActivation = 0;      // +0xe0 V  0 none, 1 relu; 2..5 "Unsupported FusedActivation type" (setFusedActivation)
	int32_t _pade4 = 0;
};
constexpr const char *kSymPoolAttrVtable = "_ZTVN4NPUC8PoolAttrE";

// ---- More layer types (dossier/npuc-harness-ops.md) ---------------------------------------------------------
// NodeLayerType: NNGraphAttribute::NNGraphAttribute(const shared_ptr<NNLayer>&) @libnpuc_frontend 0x4d8e0 compares
// NNLayer.type (case-insensitively, NPUC::iequals) with Caffe's layer names in this order and stores the index at
// NNGraphAttribute +0x30 (= NNGraphNode +0xa0); no match -> 0x43 (V, the whole chain read). NNGraphCgObject::
// addGraphNode @0x65540 dispatches on it with a jump table (V): 4/7 Input, 8 Convolution, 9/10 Pooling, 0xc Crop,
// 0xd Deconvolution, 0x12 InnerProduct, 0x18 ReLU, 0x19 ReLU6 (soc != 0), 0x1a PReLU (both -> addReLUGraphNode),
// 0x1c/0x1d/0x1f Sigmoid/TanH/Power (addActivationGraphNode), 0x26 Flatten, 0x27 Reshape, 0x2a Concat, 0x2b Slice,
// 0x2c Eltwise, 0x3c PriorBox, 0x3d Permute; every other value: "not supported" and compile error.
enum NodeLayerType : int32_t {
	LT_INPUT = 0x04, LT_CONVOLUTION = 0x08, LT_POOLING = 0x09, LT_CROP = 0x0c, LT_DECONVOLUTION = 0x0d,
	LT_INNER_PRODUCT = 0x12, LT_BATCHNORM = 0x17, LT_RELU = 0x18, LT_RELU6 = 0x19, LT_PRELU = 0x1a, LT_SIGMOID = 0x1c,
	LT_TANH = 0x1d, LT_FLATTEN = 0x26, LT_RESHAPE = 0x27, LT_CONCAT = 0x2a, LT_SLICE = 0x2b, LT_ELTWISE = 0x2c,
	LT_NONE_ACT = 0x42,  // V  the "no activation" value setFusedActivation writes at NNGraphNode +0x134
	LT_UNKNOWN = 0x43,   // V  no type matched; also the activation default of every node (NNGraphAttribute +0xc4)
};

// EltwiseAttr: no such class in NPUC or in nn-eden-drv (no typeinfo in either; the HAL never builds an Eltwise layer).
// The only reader of an "Eltwise" layer's attribute is NNGraphAttribute::getEltwiseParam @libnpuc_frontend 0x50f5c
// (V): it copies the std::string at attr +0x08 and maps "Add" -> 0, "Multiply" -> 1, "Maximum" -> 2, anything else
// -> 3, which addElemtWiseGraphNode @0x6a8c0 rejects. So ours is our own class (our vtable, LayerAttr's exported
// typeinfo as base), with that one member where NPUC reads it. No filler on the on-device chain reads an Eltwise
// node's quantisation (ZeroPoint and FusedActivation handle node types 8 and 9 only, Scale/Shift/Kernel/Bias the
// type string "Convolution" only; V @0x9e7ac, 0x9ee1c, 0x9d918...): the scales below are ours, for the record.
class NPUC_API EltwiseAttr : public LayerAttr {
public:
	std::string op;                   // +0x08 V  "Add" | "Multiply" | "Maximum" (exact case, memcmp)
	void printAttr() override {}
};

// ConcatAttr: nn-eden-drv's own class (loadConcatAttrForConcatLayer @0x175378 make_shared news 0x40 = 0x18 + 0x28,
// stores the two members below; its vtable is the HAL's, NPUC has no ConcatAttr typeinfo). Ours copies the layout.
class NPUC_API ConcatAttr : public LayerAttr {
public:
	int32_t axis = 1;                 // +0x08 V  the NNAPI axis operand (configConcatenation's first word); G: NCHW 1
	int32_t _pad0c = 0;
	std::vector<float> outScale;      // +0x10 V  the output operand's scale(s) (getOutputScaleAndZeroPoint; zps dropped)
	void printAttr() override {}
};

// An empty attribute for layers whose attribute NPUC never reads on this path (ReLU: addReLUGraphNode reads only
// the input shape; G for the rest). NNLayer::printInfo calls attr->printAttr(), so the pointer must not be null.
class NPUC_API PlainAttr : public LayerAttr {
public:
	void printAttr() override {}
};

// NNNetwork::mergeConcatLayer() (weak, exported, @libnpuc_frontend 0xb5620): what nn-eden-drv's buildNNNetwork calls
// last (V). For each "Concat" layer whose only customer is another "Concat", removeOneCustomerLayer merges them; no
// effect on a network without Concat (V, the loop only acts on type "Concat").
constexpr const char *kSymMergeConcat = "_ZN4NPUC9NNNetwork16mergeConcatLayerEv";

// NNGraphNode (0x198 bytes, operator new in NNGraph::prepareRawNodes @libnpuc_frontend 0x548c0; NPUC-internal, built
// during the compile): the few fields the Eltwise path needs, as raw offsets (V unless marked).
namespace NNGraphNodeOff {
constexpr size_t attr = 0x70;            // V  NNGraphAttribute (copy-constructed from the layer)
constexpr size_t layer = 0x78;           // V  NNGraphAttribute +0x08: NNLayer * (shared_ptr's pointer)
constexpr size_t type = 0xa0;            // V  NNGraphAttribute +0x30: NodeLayerType
constexpr size_t multipleInput = 0xe8;   // V  shared_ptr<NPUMultipleInputData>: set only by NNGraphNodeMultipleInputFiller
                                         //    (off-device chain); copied into NNGraphCgEltwiseParam by addElemtWiseGraphNode
                                         //    (@0x6aa74) and read by EltwiseNode2020::buildMetaDataDram (libnpuc_template
                                         //    0x1c2ad4), which dereferences it unchecked
constexpr size_t act = 0x134;            // V  activation NodeLayerType (setFusedActivation; 0x43 when never set)
}  // namespace NNGraphNodeOff
constexpr const char *kSymAddElemtWise = "_ZN4NPUC15NNGraphCgObject21addElemtWiseGraphNodeEPNS_11NNGraphNodeE";

// NNGraphAttribute (NNGraphNode +0x70; constructor @libnpuc_frontend 0x4d8e0): the fields the depthwise hook needs (V).
namespace NNGraphAttributeOff {
constexpr size_t layer = 0x08;           // V  NNLayer * (the shared_ptr's pointer; = NNGraphNode +0x78)
constexpr size_t framework = 0x18;       // V  std::string, "stensorflow" from the constructor (no on-device setter seen);
                                         //    DwOptSupported (@0x5333c) wants "scaffe" (memcmp, 6 bytes)
constexpr size_t type = 0x30;            // V  NodeLayerType (= NNGraphNode +0xa0)
constexpr size_t kernel = 0x38;          // V  shared_ptr to the kernel data node (= NNGraphNode +0xa8): its virtual +0x10
                                         //    returns the kernel shape {numOutput, numInput, kh, kw} (KernelONFiller)
}  // namespace NNGraphAttributeOff
// Depthwise (docs/format/depthwise.md, V): NNGraphAttribute::isDwConvolution (@0x53724) is group (ConvAttr +0x50) > 1 and
// group == C out, kernel shape [0] == [1] == C in == C out; addConvolutionGraphNode sets NN::Op::Convolution's DW flag
// (+0x79) from it, and its group (+0x84) to ConvAttr.group only when SocSupportsDwOpt() (CompileOptions.soc in 1..3),
// DwOptSupported() and not (stride h > 2 and stride w > 2), else 1. DwOptSupported: group > 1, node type 8, framework
// "scaffe", kernel shape [0] == [1], both multiples of 16.
constexpr const char *kSymDwOptSupported = "_ZN4NPUC16NNGraphAttribute14DwOptSupportedEv";

}  // namespace NPUC

// NPUMultipleInputMemoryData (global namespace, as NPUC names it): 0x1a0 bytes (make_shared of 0x1b8 in
// NNGraphNodeMultipleInputFiller::setMultipleInput @libnpuc_frontend 0xa46e4, memset of 0x198 after the vptr; its D2 @0xb2690
// frees the 17 vectors below). Vtable (V, relocations at 0xed5f8): [~D2, ~D0, getInputData(unsigned) @0xb27f8 = +0x08 +
// i * 0x18, getFractionalLengthData(unsigned) @0xb2808 = +0xc8 + i * 0x18], exported (_ZTV26NPUMultipleInputMemoryData,
// weak); the destructor is declared and never defined here, so the compiler takes NPUC's vtable (as for ConvAttr).
// What setMultipleInput stores (V): fl[0] = the node's input 0 fractional lengths, fl[1] = input 1's, fl[8] = the output's,
// one byte per output channel (getInFractionalLength(0), (1), getOutFractionalLength() of the node's NPUFractionalLength);
// buildMetaDataDram copies fl[0][c], fl[1][c], fl[8][c] into the Eltwise meta data blob (V, 0x1c2d10..0x1c2d88): the NPU
// receives power-of-two exponents (G: out = in1 * 2^(flo - fl1) + in2 * 2^(flo - fl2), rounded and saturated).
class NPUC_API NPUMultipleInputMemoryData {
public:
	NPUMultipleInputMemoryData() {}
	virtual ~NPUMultipleInputMemoryData();
	virtual std::vector<uint8_t> &getInputData(unsigned i);
	virtual std::vector<uint8_t> &getFractionalLengthData(unsigned i);

	std::vector<uint8_t> inputData[8];    // +0x08 O  empty on the memory path (the file variant reads blobs into them)
	std::vector<uint8_t> fl[9];           // +0xc8 V  per-channel fractional lengths: [0] input 1, [1] input 2, [8] output
};

namespace NPUC {

// ---- NNLayer (0xa8 bytes, V: operator new(0xa8) for NNConvLayer and NNInputLayer; ctor @libnpuc_frontend
// 0x90b7c). Vtable (V): [~D2, ~D0, computeOutputShape(const vector<vector<int>>&)]. Only the base-object
// constructors (C2) are exported, so a harness calls them by symbol on raw storage (npuc_new_layer below). -------
class NPUC_API NNLayer {
public:
	virtual ~NNLayer();
	virtual void computeOutputShape(const std::vector<std::vector<int32_t>> &inShapes);

	std::string name;                              // +0x08 V
	std::string type;                              // +0x20 V  Caffe names: "Input", "Convolution", "Pooling"...
	std::vector<std::string> inputs;               // +0x38 V  bottom blobs ("ROOT" for an Input layer)
	std::vector<std::string> outputs;              // +0x50 V  top blobs
	std::vector<std::string> customers;            // +0x68 O  (removeCustomer)
	std::map<std::string, std::vector<int32_t>> shapes;  // +0x80 V  blob name -> NCHW shape, inputs and outputs.
	//   Filled by the builder, not by NPUC: the HAL calls computeOutputShape(input shapes) on each layer as it
	//   creates it. getOutputShape(i) (@frontend 0x52e30) asserts shapes[outputs[i]] has 2 or 4 dims ("tfShape"),
	//   copies it as is when CompileOptions +0x13c is 0 (the HAL's NCHW case), else reorders it.
	std::shared_ptr<LayerAttr> attr;               // +0x98 V
};
// NNConvLayer: NNLayer + its own vtable (computeOutputShape from pad/stride/dilation/kernel/numOutput), no member
// (V: its D2 is NNLayer's). NNInputLayer exists only in nn-eden-drv: there it is an NNLayer of type "Input", and
// NPUC's computeLayersShape skips layers of type "Input" (V @0xb5018): the base NNLayer is enough for it.
class NPUC_API NNConvLayer : public NNLayer {
public:
	void computeOutputShape(const std::vector<std::vector<int32_t>> &inShapes) override;
};

// NNPoolLayer: NNLayer + its own vtable (computeOutputShape from PoolAttr's pad/stride/kernel), no member (V: operator
// new(0xa8) in nn-eden-drv's getPoolLayer @0x16ec3c; Caffe type "Pooling"). Its C2 constructor is exported.
class NPUC_API NNPoolLayer : public NNLayer {
public:
	void computeOutputShape(const std::vector<std::vector<int32_t>> &inShapes) override;
};

// The exported base-object constructors (C2). Itanium ABI: the by-value class arguments are passed as pointers to
// caller-owned temporaries, destroyed by the caller after the call (the callee copies them: V in both ctors).
constexpr const char *kSymNNLayerC2 =
	"_ZN4NPUC7NNLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_NS5_"
	"IS7_EEEESA_NS1_10shared_ptrINS_9LayerAttrEEE";
constexpr const char *kSymNNConvLayerC2 =
	"_ZN4NPUC11NNConvLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_"
	"NS5_IS7_EEEESA_NS1_10shared_ptrINS_8ConvAttrEEE";
constexpr const char *kSymNNPoolLayerC2 =
	"_ZN4NPUC11NNPoolLayerC2ENSt6__ndk112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_NS1_6vectorIS7_"
	"NS5_IS7_EEEESA_NS1_10shared_ptrINS_8PoolAttrEEE";
using NNLayerC2 = void (*)(NNLayer *self, std::string name, std::string type, std::vector<std::string> inputs,
                           std::vector<std::string> outputs, std::shared_ptr<LayerAttr> attr);
using NNConvLayerC2 = void (*)(NNConvLayer *self, std::string name, std::string type, std::vector<std::string> inputs,
                               std::vector<std::string> outputs, std::shared_ptr<ConvAttr> attr);

// ---- NNNetwork (0x60 bytes, V: make_shared of 0x78 in CompilerManager::compile and buildNNNetwork; destructor
// @libnpuc_frontend 0xb3f80, exported). No vptr. ----------------------------------------------------------------
struct NNNetwork {
	std::vector<std::shared_ptr<NNLayer>> layers;                    // +0x00 V  in append order, then sorted
	std::map<std::string, std::vector<int32_t>> shapes;              // +0x18 O  (destroyed last; filler unknown)
	std::map<std::string, std::shared_ptr<NNLayer>> layerByName;     // +0x30 V  appendLayers: [layer.name]
	std::map<std::string, std::shared_ptr<NNLayer>> producerOf;      // +0x48 V  appendLayers: [each output]
};
// NNNetwork::appendLayers(NNLayer*) is inline in nn-eden-drv (@0x6e758) and not exported: npuc_append below is
// the same steps (duplicate name -> 7, else push_back + both maps -> 0).
constexpr const char *kSymNNNetworkD1 = "_ZN4NPUC9NNNetworkD1Ev";

inline int npuc_append(NNNetwork &net, const std::shared_ptr<NNLayer> &l)
{
	for (auto &e : net.layers)
		if (e->name == l->name)
			return 7;
	net.layers.push_back(l);
	net.layerByName[l->name] = l;
	for (auto &o : l->outputs)
		net.producerOf[o] = l;
	return 0;
}

}  // namespace NPUC

static_assert(sizeof(std::string) == 0x18 && sizeof(std::vector<int>) == 0x18, "libc++ ABI v1");
static_assert(sizeof(std::map<std::string, int>) == 0x18 && sizeof(std::shared_ptr<int>) == 0x10, "libc++ ABI v1");
static_assert(offsetof(NPUC::ConvAttr, pad) == 0x08 && offsetof(NPUC::ConvAttr, group) == 0x50, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, kernel) == 0x60 && offsetof(NPUC::ConvAttr, attrType) == 0x78, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, padType) == 0x90 && offsetof(NPUC::ConvAttr, kernelName) == 0xa8, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, biasName) == 0xc0 && offsetof(NPUC::ConvAttr, hasBias) == 0xd8, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, data0) == 0xe0 && offsetof(NPUC::ConvAttr, kernelData) == 0x100, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, biasData) == 0x120 && offsetof(NPUC::ConvAttr, inScale) == 0x130, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, outScale) == 0x160 && offsetof(NPUC::ConvAttr, wScale) == 0x190, "ConvAttr");
static_assert(offsetof(NPUC::ConvAttr, bZp) == 0x1d8 && offsetof(NPUC::ConvAttr, fusedActivation) == 0x1f0, "ConvAttr");
static_assert(sizeof(NPUC::ConvAttr) == 0x1f8, "ConvAttr");
static_assert(offsetof(NPUC::NNLayer, name) == 0x08 && offsetof(NPUC::NNLayer, inputs) == 0x38, "NNLayer");
static_assert(offsetof(NPUC::NNLayer, shapes) == 0x80 && offsetof(NPUC::NNLayer, attr) == 0x98, "NNLayer");
static_assert(sizeof(NPUC::NNLayer) == 0xa8 && sizeof(NPUC::NNConvLayer) == 0xa8, "NNLayer");
static_assert(offsetof(NPUC::NNNetwork, layerByName) == 0x30 && sizeof(NPUC::NNNetwork) == 0x60, "NNNetwork");
static_assert(offsetof(NPUC::PoolAttr, pad) == 0x08 && offsetof(NPUC::PoolAttr, stride) == 0x20, "PoolAttr");
static_assert(offsetof(NPUC::PoolAttr, kernel) == 0x38 && offsetof(NPUC::PoolAttr, padType) == 0x50, "PoolAttr");
static_assert(offsetof(NPUC::PoolAttr, poolType) == 0x68 && offsetof(NPUC::PoolAttr, inScale) == 0x80, "PoolAttr");
static_assert(offsetof(NPUC::PoolAttr, inZp) == 0x98 && offsetof(NPUC::PoolAttr, outScale) == 0xb0, "PoolAttr");
static_assert(offsetof(NPUC::PoolAttr, outZp) == 0xc8 && offsetof(NPUC::PoolAttr, fusedActivation) == 0xe0, "PoolAttr");
static_assert(sizeof(NPUC::PoolAttr) == 0xe8 && sizeof(NPUC::NNPoolLayer) == 0xa8, "PoolAttr");
static_assert(offsetof(NPUC::EltwiseAttr, op) == 0x08 && sizeof(NPUC::EltwiseAttr) == 0x20, "EltwiseAttr");
static_assert(offsetof(NPUC::ConcatAttr, axis) == 0x08 && offsetof(NPUC::ConcatAttr, outScale) == 0x10, "ConcatAttr");
static_assert(sizeof(NPUC::ConcatAttr) == 0x28, "ConcatAttr");
static_assert(offsetof(NPUMultipleInputMemoryData, inputData) == 0x08, "NPUMultipleInputMemoryData");
static_assert(offsetof(NPUMultipleInputMemoryData, fl) == 0xc8, "NPUMultipleInputMemoryData");
static_assert(sizeof(NPUMultipleInputMemoryData) == 0x1a0, "NPUMultipleInputMemoryData");
#pragma clang diagnostic pop
