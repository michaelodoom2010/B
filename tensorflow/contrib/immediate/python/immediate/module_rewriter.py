"""Contains ModuleRewriter class + helper methods useful for writing custom
symbol_rewriter functions to be used with ModuleRewriter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import re
import sys
import types

from .op import OpWrapper
from .op import OpDefLibraryWrapper
from .op import ConvertToTensorWrapper
from .op import ConstantOpWrapper

from .tensor import _ENABLE_DEBUG_LOGGING

__all__ = ["ModuleRewriter"]

def get_symbol_file(symbol):
  """Returns filename of symbol definition, empty string if not available."""

  if hasattr(symbol, "__file__"):
    return symbol.__file__
  elif not isinstance(symbol, types.ModuleType):
    try:
      symbol_module = sys.modules[symbol.__module__]
      return symbol_module.__file__
    except (AttributeError, KeyError):
      return ""


def get_symbol_name(symbol):
  """Returns __name__ attribute or empty string if not available."""
  if hasattr(symbol, "__name__"):
    return symbol.__name__
  else:
    return ""


def copy_function(old_func, updated_module):
  """Copies a function, updating it to point to given module."""

  # Decorators don't set __module__ to the current function
  # detect this case and return it unchanged
  if old_func.__globals__ != sys.modules[old_func.__module__].__dict__:
    return old_func
    
  new_func = types.FunctionType(old_func.__code__, updated_module.__dict__,
                                name=old_func.__name__,
                                argdefs=old_func.__defaults__,
                                closure=old_func.__closure__)
  new_func.__dict__.update(old_func.__dict__)
  new_func.__module__ = updated_module.__name__
  return new_func


class ModuleRewriter(object):
  """Object that controls rewriting of module."""

  def __init__(self, symbol_rewriter, module_prefix="newmodule."):
    """Initialize ModuleRewriter.

    Rewriting is done by taking a custom "symbol-rewriter" and applying it to
    all symbols referenced from module provided to later __call__, directly or
    indirectly. If symbol-rewriter returns non-None value, the entire module
    is copied, the affected symbol is replaced with output of symbol rewriter
    and all references to old module are updated to point to new module. If
    module is not affected by symbol-rewriter, original reference is returned.

    Only function/module references are followed. This means that while object
    and type references are retained in new module, their respective references
    are not updated and they will continue point to the old module hierarchy.

    Args:
      symbol_rewriter: callable object that implements symbol rewriting. It
          should accepts a symbol (ie, a function) and return new symbol that
          acts as a replacement, or None to keep original symbol unchanged.
          The name of the symbol should remain unchanged because it's used
          to resolve references from other modules.
      module_prefix: a string that is prefixed to __name__ and __file__
          attributes of copied modules. Because we add new modules to
          sys.modules, this string must be non-empty.
    """

    assert module_prefix, "Module prefix must be non-empty"

    self.symbol_rewriter = symbol_rewriter
    self.module_prefix = module_prefix

    self._done_modules = {}  # dict of old_module->new_module
    self._module_stack = []  # stack of modules to detect cycles


  def __call__(self, original_module):
    return self._rewrite_module(original_module)

  def _rewrite_module(self, original_module):
    """Apply symbol_rewriter to given module and its dependencies recursively
    and return the result. Copies of objects are made as necessary and original
    module remains unchanged.

    Args:
      original_module: module to rewrite.

    Returns:
      Copy of module hierarchy with rewritten symbols.
    """

    # system modules are missing __file__ attribute, and checking by
    # id is insufficient to prevent infinite loops, hence forbid missing
    # __file__
    if not hasattr(original_module, "__file__") or not original_module.__file__:
      self._done_modules[original_module] = original_module

    if original_module in self._done_modules:
      return self._done_modules[original_module]

    #    self._module_stack.append(get_symbol_file(original_module))
    self._module_stack.append(original_module.__file__)
    updated_symbols = {}  # symbols that got touched


    # Go over all symbols in a module to determine if module needs to be copied
    for symbol_name, symbol in original_module.__dict__.items():

      # Case 1: symbol is directly replaced by symbol_rewriter
      new_symbol = self.symbol_rewriter(symbol)
      if new_symbol:
        updated_symbols[symbol_name] = new_symbol
        if _ENABLE_DEBUG_LOGGING:
          print("Rewrote symbol %s in %s" % (symbol_name,
                                             original_module.__name__))

      # Case 2: symbol is a module which may be affected by symbol_rewriter
      elif isinstance(symbol, types.ModuleType):
        if get_symbol_file(symbol) not in self._module_stack:
          new_symbol = self._rewrite_module(symbol)

          if new_symbol.__name__ != symbol.__name__:
            updated_symbols[symbol_name] = new_symbol

      # Case 3: symbol is a function defined in a module which may be affected
      # by symbol rewriter
      elif hasattr(symbol, "__module__") and isinstance(symbol,
                                                        types.FunctionType):
        if symbol.__module__ != original_module.__name__:
          symbol_file = get_symbol_file(symbol)
          if symbol_file and symbol_file not in self._module_stack:
            symbol_module = sys.modules[symbol.__module__]
            new_symbol_module = self._rewrite_module(symbol_module)

            if new_symbol_module.__name__ != symbol_module.__name__:
              updated_symbols[symbol_name] = new_symbol_module.__dict__[
                  symbol.__name__]

    # nothing was modified, so return module unchanged
    if not updated_symbols:
      self._done_modules[original_module] = original_module
      self._module_stack.pop()
      return original_module

    # module was modified, hence make a new copy
    new_module_name = self.module_prefix + original_module.__name__
    new_module = imp.new_module(new_module_name)
    new_module.__package__ = ""
    new_module.__file__ = self.module_prefix + original_module.__file__
    #    print("Creating module: ", new_module.__file__)
    for symbol_name, symbol in original_module.__dict__.items():

      # don't rewrite new module attributes that we just set
      if symbol_name in ('__file__', '__name__', '__package__'):
        continue

      if symbol_name in updated_symbols:
        new_symbol = updated_symbols[symbol_name]
        if (hasattr(new_symbol, "__module__") and
            new_symbol.__module__ == original_module.__name__):
          new_symbol.__module__ = new_module.__name__

        new_module.__dict__[symbol_name] = new_symbol

      # it's a function whose definition wasn't updated
      elif isinstance(symbol, types.FunctionType):
        # if it's a function in current module, copy it to update its globals
        if symbol.__module__ == original_module.__name__:
          new_symbol = copy_function(symbol, new_module)
        # otherwise retain old reference
        else:
          new_symbol = symbol
        new_module.__dict__[symbol_name] = new_symbol

      else:  # objects, classes, constants remain unchanged
        new_module.__dict__[symbol_name] = symbol

    sys.modules[new_module_name] = new_module
    self._done_modules[original_module] = new_module
    self._module_stack.pop()
    return new_module

class AddSymbolRewriter(object):
  """An object implementing simple symbol rewriter."""

  def __init__(self, value):
    def new_add(arg1, arg2):
      return value

    self.replacement_func = new_add

  def __call__(self, symbol):
    if (get_symbol_name(symbol) == "add" and
        "gen_math_ops.py" in get_symbol_file(symbol)):
        return self.replacement_func


from tensorflow.python.ops import op_def_library
class OpDefLibRewriter(object):
  """Replaces op_def_lib in gen_ops files with custom version."""
  
  # make this a formal dependency
  def __init__(self, env):
    self.env = env
    self._fname_re = re.compile(".*tensorflow/python/ops/gen_.*_ops.pyc?$")
    def filematch(symbol):
      fn = get_symbol_file(symbol)
      return bool(self._fname_re.findall(fn))
    self.file_matches = filematch


  def __call__(self, symbol):
    # TODO(yaroslavvb): add filename filtering?
    if (isinstance(symbol, op_def_library.OpDefLibrary)):
      return OpDefLibraryWrapper(self.env, symbol)

# TODO: add optional filename argument for rewriter?
class ImmediateRewriter(object):
  """Replaces all relevant symbols with corresponding immediate versions."""
  
  def __init__(self, env):
    self.env = env

  def __call__(self, symbol):
    # replace _op_lib_def in gen_.*_ops files
    if isinstance(symbol, op_def_library.OpDefLibrary):
      return OpDefLibraryWrapper(self.env, symbol)

    if isinstance(symbol, types.FunctionType):
      if (symbol.__name__ == 'convert_to_tensor' and
          symbol.__module__ == 'tensorflow.python.framework.ops'):
        return ConvertToTensorWrapper(self.env, symbol)

      if (symbol.__name__ == 'constant' and
          symbol.__module__=='tensorflow.python.ops.constant_op'):
        return ConstantOpWrapper(self.env, symbol)
        
