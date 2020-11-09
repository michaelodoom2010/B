// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/protobuf/critical_section.proto

package for_core_protos_go_proto

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

// Protocol buffer representing a CriticalSection.
type CriticalSectionDef struct {
	// Name of the critical section handle.
	CriticalSectionName  string   `protobuf:"bytes,1,opt,name=critical_section_name,json=criticalSectionName,proto3" json:"critical_section_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *CriticalSectionDef) Reset()         { *m = CriticalSectionDef{} }
func (m *CriticalSectionDef) String() string { return proto.CompactTextString(m) }
func (*CriticalSectionDef) ProtoMessage()    {}
func (*CriticalSectionDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_d30d8be90fd098b9, []int{0}
}

func (m *CriticalSectionDef) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CriticalSectionDef.Unmarshal(m, b)
}
func (m *CriticalSectionDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CriticalSectionDef.Marshal(b, m, deterministic)
}
func (m *CriticalSectionDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CriticalSectionDef.Merge(m, src)
}
func (m *CriticalSectionDef) XXX_Size() int {
	return xxx_messageInfo_CriticalSectionDef.Size(m)
}
func (m *CriticalSectionDef) XXX_DiscardUnknown() {
	xxx_messageInfo_CriticalSectionDef.DiscardUnknown(m)
}

var xxx_messageInfo_CriticalSectionDef proto.InternalMessageInfo

func (m *CriticalSectionDef) GetCriticalSectionName() string {
	if m != nil {
		return m.CriticalSectionName
	}
	return ""
}

// Protocol buffer representing a CriticalSection execution.
type CriticalSectionExecutionDef struct {
	// Name of the critical section handle.
	ExecuteInCriticalSectionName string `protobuf:"bytes,1,opt,name=execute_in_critical_section_name,json=executeInCriticalSectionName,proto3" json:"execute_in_critical_section_name,omitempty"`
	// Whether this operation requires exclusive access to its resources,
	// (i.e., no other CriticalSections may request the same resources).
	ExclusiveResourceAccess bool     `protobuf:"varint,2,opt,name=exclusive_resource_access,json=exclusiveResourceAccess,proto3" json:"exclusive_resource_access,omitempty"`
	XXX_NoUnkeyedLiteral    struct{} `json:"-"`
	XXX_unrecognized        []byte   `json:"-"`
	XXX_sizecache           int32    `json:"-"`
}

func (m *CriticalSectionExecutionDef) Reset()         { *m = CriticalSectionExecutionDef{} }
func (m *CriticalSectionExecutionDef) String() string { return proto.CompactTextString(m) }
func (*CriticalSectionExecutionDef) ProtoMessage()    {}
func (*CriticalSectionExecutionDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_d30d8be90fd098b9, []int{1}
}

func (m *CriticalSectionExecutionDef) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CriticalSectionExecutionDef.Unmarshal(m, b)
}
func (m *CriticalSectionExecutionDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CriticalSectionExecutionDef.Marshal(b, m, deterministic)
}
func (m *CriticalSectionExecutionDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CriticalSectionExecutionDef.Merge(m, src)
}
func (m *CriticalSectionExecutionDef) XXX_Size() int {
	return xxx_messageInfo_CriticalSectionExecutionDef.Size(m)
}
func (m *CriticalSectionExecutionDef) XXX_DiscardUnknown() {
	xxx_messageInfo_CriticalSectionExecutionDef.DiscardUnknown(m)
}

var xxx_messageInfo_CriticalSectionExecutionDef proto.InternalMessageInfo

func (m *CriticalSectionExecutionDef) GetExecuteInCriticalSectionName() string {
	if m != nil {
		return m.ExecuteInCriticalSectionName
	}
	return ""
}

func (m *CriticalSectionExecutionDef) GetExclusiveResourceAccess() bool {
	if m != nil {
		return m.ExclusiveResourceAccess
	}
	return false
}

func init() {
	proto.RegisterType((*CriticalSectionDef)(nil), "tensorflow.CriticalSectionDef")
	proto.RegisterType((*CriticalSectionExecutionDef)(nil), "tensorflow.CriticalSectionExecutionDef")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/critical_section.proto", fileDescriptor_d30d8be90fd098b9)
}

var fileDescriptor_d30d8be90fd098b9 = []byte{
	// 260 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x51, 0xbd, 0x4e, 0xc3, 0x30,
	0x10, 0x96, 0x19, 0x10, 0x78, 0x0c, 0xaa, 0x08, 0x82, 0x21, 0xea, 0xd4, 0x29, 0x91, 0x60, 0x63,
	0xa3, 0x05, 0x04, 0x0b, 0xaa, 0x82, 0x58, 0x58, 0xac, 0xe4, 0x74, 0x0e, 0x16, 0x89, 0x0f, 0x9d,
	0x6d, 0xda, 0x97, 0xe0, 0x21, 0x78, 0x4b, 0x46, 0x54, 0x37, 0x50, 0x08, 0xa8, 0xdb, 0xe7, 0xfb,
	0xfe, 0xa4, 0xcf, 0xb2, 0xf0, 0x68, 0x1d, 0xb1, 0x6e, 0x69, 0x51, 0x00, 0x31, 0x16, 0x2f, 0x4c,
	0x9e, 0xea, 0xa0, 0x0b, 0x60, 0xe3, 0x0d, 0x54, 0xad, 0x72, 0x08, 0xde, 0x90, 0xcd, 0x23, 0x93,
	0xc8, 0x8d, 0x61, 0x7c, 0x23, 0x93, 0x59, 0xaf, 0xba, 0x5f, 0x8b, 0x2e, 0x51, 0x27, 0xa7, 0x72,
	0x34, 0xf4, 0x2a, 0x5b, 0x75, 0x98, 0x8a, 0x4c, 0x4c, 0xf6, 0xcb, 0x03, 0xf8, 0x6d, 0xb9, 0xab,
	0x3a, 0x1c, 0xbf, 0x0b, 0x79, 0x3c, 0x88, 0xba, 0x5a, 0x22, 0x84, 0xaf, 0xcc, 0x6b, 0x99, 0x61,
	0x7c, 0xa3, 0x32, 0x56, 0x6d, 0x8b, 0x3f, 0xe9, 0x75, 0xb7, 0x76, 0xf6, 0xb7, 0x27, 0x39, 0x97,
	0x47, 0xb8, 0x84, 0x36, 0x38, 0xf3, 0x8a, 0x8a, 0xd1, 0x51, 0x60, 0x40, 0x55, 0x01, 0xa0, 0x73,
	0xe9, 0x4e, 0x26, 0x26, 0x7b, 0xe5, 0xe1, 0xb7, 0xa0, 0xec, 0xf9, 0x8b, 0x48, 0x4f, 0xdf, 0x84,
	0x4c, 0x89, 0x9b, 0x7c, 0x33, 0x40, 0xae, 0xb9, 0xea, 0x70, 0x41, 0xfc, 0x3c, 0x1d, 0x0d, 0xda,
	0xe6, 0xab, 0xb1, 0xdc, 0x5c, 0x3c, 0x3e, 0x34, 0xc6, 0x3f, 0x85, 0x3a, 0x07, 0xea, 0x7e, 0x6e,
	0xfd, 0x3f, 0x6c, 0x68, 0xf0, 0x09, 0x9a, 0x58, 0xad, 0x2e, 0x2a, 0x5e, 0x9c, 0x6a, 0x68, 0x8d,
	0x3e, 0x84, 0xa8, 0x77, 0x23, 0x3a, 0xfb, 0x0c, 0x00, 0x00, 0xff, 0xff, 0xfa, 0xdf, 0x90, 0x13,
	0xc3, 0x01, 0x00, 0x00,
}
