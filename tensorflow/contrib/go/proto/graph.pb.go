// Code generated by protoc-gen-go.
// source: tensorflow/core/framework/graph.proto
// DO NOT EDIT!

package tensorflow

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// Represents the graph of operations
type GraphDef struct {
	Node []*NodeDef `protobuf:"bytes,1,rep,name=node" json:"node,omitempty"`
	// Compatibility versions of the graph.  See core/public/version.h for version
	// history.  The GraphDef version is distinct from the TensorFlow version, and
	// each release of TensorFlow will support a range of GraphDef versions.
	Versions *VersionDef `protobuf:"bytes,4,opt,name=versions" json:"versions,omitempty"`
	// Deprecated single version field; use versions above instead.  Since all
	// GraphDef changes before "versions" was introduced were forward
	// compatible, this field is entirely ignored.
	Version int32 `protobuf:"varint,3,opt,name=version" json:"version,omitempty"`
	// EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
	//
	// "library" provides user-defined functions.
	//
	// Naming:
	//   * library.function.name are in a flat namespace.
	//     NOTE: We may need to change it to be hierarchical to support
	//     different orgs. E.g.,
	//     { "/google/nn", { ... }},
	//     { "/google/vision", { ... }}
	//     { "/org_foo/module_bar", {...}}
	//     map<string, FunctionDefLib> named_lib;
	//   * If node[i].op is the name of one function in "library",
	//     node[i] is deemed as a function call. Otherwise, node[i].op
	//     must be a primitive operation supported by the runtime.
	//
	//
	// Function call semantics:
	//
	//   * The callee may start execution as soon as some of its inputs
	//     are ready. The caller may want to use Tuple() mechanism to
	//     ensure all inputs are ready in the same time.
	//
	//   * The consumer of return values may start executing as soon as
	//     the return values the consumer depends on are ready.  The
	//     consumer may want to use Tuple() mechanism to ensure the
	//     consumer does not start until all return values of the callee
	//     function are ready.
	Library *FunctionDefLibrary `protobuf:"bytes,2,opt,name=library" json:"library,omitempty"`
}

func (m *GraphDef) Reset()                    { *m = GraphDef{} }
func (m *GraphDef) String() string            { return proto.CompactTextString(m) }
func (*GraphDef) ProtoMessage()               {}
func (*GraphDef) Descriptor() ([]byte, []int) { return fileDescriptor4, []int{0} }

func (m *GraphDef) GetNode() []*NodeDef {
	if m != nil {
		return m.Node
	}
	return nil
}

func (m *GraphDef) GetVersions() *VersionDef {
	if m != nil {
		return m.Versions
	}
	return nil
}

func (m *GraphDef) GetLibrary() *FunctionDefLibrary {
	if m != nil {
		return m.Library
	}
	return nil
}

type NodeDef struct {
	// The name given to this operator. Used for naming inputs,
	// logging, visualization, etc.  Unique within a single GraphDef.
	// Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_./]*".
	Name string `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	// The operation name.  There may be custom parameters in attrs.
	// Op names starting with an underscore are reserved for internal use.
	Op string `protobuf:"bytes,2,opt,name=op" json:"op,omitempty"`
	// Each input is "node:src_output" with "node" being a string name and
	// "src_output" indicating which output tensor to use from "node". If
	// "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
	// may optionally be followed by control inputs that have the format
	// "^node".
	Input []string `protobuf:"bytes,3,rep,name=input" json:"input,omitempty"`
	// A (possibly partial) specification for the device on which this
	// node should be placed.
	// The expected syntax for this string is as follows:
	//
	// DEVICE_SPEC ::= COLOCATED_NODE | PARTIAL_SPEC
	//
	// COLOCATED_NODE ::= "@" NODE_NAME  // See NodeDef.name above.
	// PARTIAL_SPEC ::= ("/" CONSTRAINT) *
	// CONSTRAINT ::= ("job:" JOB_NAME)
	//              | ("replica:" [1-9][0-9]*)
	//              | ("task:" [1-9][0-9]*)
	//              | ( ("gpu" | "cpu") ":" ([1-9][0-9]* | "*") )
	//
	// Valid values for this string include:
	// * "@other/node"                         (colocate with "other/node")
	// * "/job:worker/replica:0/task:1/gpu:3"  (full specification)
	// * "/job:worker/gpu:3"                   (partial specification)
	// * ""                                    (no specification)
	//
	// If the constraints do not resolve to a single device (or if this
	// field is empty or not present), the runtime will attempt to
	// choose a device automatically.
	Device string `protobuf:"bytes,4,opt,name=device" json:"device,omitempty"`
	// Operation-specific graph-construction-time configuration.
	// Note that this should include all attrs defined in the
	// corresponding OpDef, including those with a value matching
	// the default -- this allows the default to change and makes
	// NodeDefs easier to interpret on their own.  However, if
	// an attr with a default is not specified in this list, the
	// default will be used.
	// The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
	// one of the names from the corresponding OpDef's attr field).
	// The values must have a type matching the corresponding OpDef
	// attr's type field.
	// TODO(josh11b): Add some examples here showing best practices.
	Attr map[string]*AttrValue `protobuf:"bytes,5,rep,name=attr" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
}

func (m *NodeDef) Reset()                    { *m = NodeDef{} }
func (m *NodeDef) String() string            { return proto.CompactTextString(m) }
func (*NodeDef) ProtoMessage()               {}
func (*NodeDef) Descriptor() ([]byte, []int) { return fileDescriptor4, []int{1} }

func (m *NodeDef) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

func init() {
	proto.RegisterType((*GraphDef)(nil), "tensorflow.GraphDef")
	proto.RegisterType((*NodeDef)(nil), "tensorflow.NodeDef")
}

var fileDescriptor4 = []byte{
	// 358 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0x84, 0x91, 0xcf, 0x4e, 0xc2, 0x40,
	0x10, 0xc6, 0xb3, 0x2d, 0x05, 0x3a, 0x24, 0xc6, 0xac, 0x4a, 0x1a, 0xa2, 0x86, 0x90, 0x18, 0x51,
	0x93, 0x12, 0xf1, 0x42, 0xbc, 0x49, 0xfc, 0x73, 0x31, 0x84, 0xec, 0x81, 0xab, 0x29, 0xb0, 0xc5,
	0x06, 0xec, 0x36, 0xdb, 0x05, 0xc2, 0xd3, 0xf9, 0x26, 0x3e, 0x8b, 0xb3, 0xdb, 0x02, 0x3d, 0x48,
	0xbc, 0xcd, 0xce, 0xfc, 0xe6, 0xcb, 0x7c, 0xfb, 0xc1, 0x95, 0xe2, 0x71, 0x2a, 0x64, 0xb8, 0x10,
	0xeb, 0xce, 0x44, 0x48, 0xde, 0x09, 0x65, 0xf0, 0xc5, 0xd7, 0x42, 0xce, 0x3b, 0x33, 0x19, 0x24,
	0x9f, 0x7e, 0x22, 0x85, 0x12, 0x14, 0xf6, 0x58, 0xe3, 0xf6, 0xf0, 0x4a, 0xa0, 0x94, 0xfc, 0x58,
	0x05, 0x8b, 0x25, 0xcf, 0xf6, 0x1a, 0xed, 0xc3, 0x6c, 0xb8, 0x8c, 0x27, 0x2a, 0x12, 0xf1, 0xff,
	0xe4, 0x8a, 0xcb, 0x14, 0xc1, 0x34, 0x23, 0x5b, 0xdf, 0x04, 0xaa, 0x6f, 0xfa, 0xb6, 0x67, 0x1e,
	0xd2, 0x6b, 0x28, 0xc5, 0x62, 0xca, 0x3d, 0xd2, 0xb4, 0xdb, 0xb5, 0xee, 0x89, 0xbf, 0x57, 0xf1,
	0x07, 0xd8, 0x47, 0x84, 0x19, 0x80, 0x76, 0xa1, 0xba, 0xd5, 0xf1, 0x4a, 0x4d, 0x82, 0x70, 0xbd,
	0x08, 0x8f, 0xb2, 0x99, 0xe6, 0x77, 0x1c, 0x3d, 0x87, 0x4a, 0x5e, 0x7b, 0x36, 0xae, 0x38, 0x7d,
	0xcb, 0x23, 0x6c, 0xdb, 0xa2, 0x3d, 0xa8, 0x2c, 0xa2, 0xb1, 0x0c, 0xe4, 0xc6, 0xb3, 0x8c, 0xe0,
	0x65, 0x51, 0xf0, 0x35, 0xb7, 0x87, 0x8a, 0xef, 0x19, 0xc5, 0xb6, 0x78, 0xeb, 0x87, 0x40, 0x25,
	0xbf, 0x8e, 0x52, 0x34, 0x80, 0x46, 0xd1, 0x00, 0x69, 0xbb, 0xcc, 0xd4, 0xf4, 0x08, 0x2c, 0x91,
	0x18, 0x51, 0x97, 0x61, 0x45, 0x4f, 0xc1, 0x89, 0xe2, 0x64, 0xa9, 0xf0, 0x0a, 0x1b, 0x5b, 0xd9,
	0x83, 0xd6, 0xa1, 0x3c, 0xe5, 0xab, 0x68, 0xc2, 0x8d, 0x1f, 0x97, 0xe5, 0x2f, 0x7a, 0x0f, 0x25,
	0x9d, 0x83, 0xe7, 0x98, 0x2f, 0xb9, 0xf8, 0xe3, 0x4b, 0xfc, 0x27, 0x9c, 0xbf, 0xc4, 0x0a, 0x6f,
	0x32, 0x68, 0x63, 0x00, 0xee, 0xae, 0x45, 0x8f, 0xc1, 0x9e, 0xf3, 0x4d, 0x7e, 0x90, 0x2e, 0xe9,
	0x1d, 0x38, 0x26, 0xd4, 0xdc, 0xe7, 0x59, 0x51, 0x52, 0xef, 0x8d, 0xf4, 0x90, 0x65, 0xcc, 0xa3,
	0xd5, 0x23, 0xfd, 0x1b, 0xf0, 0x84, 0x9c, 0x15, 0xb1, 0x5d, 0x9a, 0xfd, 0x9a, 0xc9, 0x6e, 0xa8,
	0xa3, 0x4c, 0x87, 0x64, 0x5c, 0x36, 0xa1, 0x3e, 0xfc, 0x06, 0x00, 0x00, 0xff, 0xff, 0x96, 0xa5,
	0x75, 0xb8, 0x89, 0x02, 0x00, 0x00,
}
