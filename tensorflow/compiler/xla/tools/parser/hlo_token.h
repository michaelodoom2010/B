/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_TOKEN_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_TOKEN_H_

namespace xla {
namespace tools {

// Defines different kinds of tokens in a hlo module string.
enum class TokKind {
  // Markers
  kEof,
  kError,

  // Tokens with no info.
  kEqual,  // =
  kComma,  // ,
  kColon,  // :
  kLsquare,
  kRsquare,  // [  ]
  kLbrace,
  kRbrace,  // {  }
  kLparen,
  kRparen,  // (  )

  kArrow,  // ->

  // Keywords
  kw_HloModule,
  kw_ENTRY,
  kw_true,
  kw_false,

  // Typed tokens.
  kName,     // %foo
  kShape,    // f32[2,3]{1,0}
  kOpcode,   // add
  kInt,      // 42
  kDecimal,  // 4.2
};

}  // namespace tools
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_TOKEN_H_
