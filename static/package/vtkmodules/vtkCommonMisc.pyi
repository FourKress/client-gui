from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore

VTK_PARSER_ABSOLUTE_VALUE:int
VTK_PARSER_ADD:int
VTK_PARSER_AND:int
VTK_PARSER_ARCCOSINE:int
VTK_PARSER_ARCSINE:int
VTK_PARSER_ARCTANGENT:int
VTK_PARSER_BEGIN_VARIABLES:int
VTK_PARSER_CEILING:int
VTK_PARSER_COSINE:int
VTK_PARSER_CROSS:int
VTK_PARSER_DIVIDE:int
VTK_PARSER_DOT_PRODUCT:int
VTK_PARSER_EQUAL_TO:int
VTK_PARSER_ERROR_RESULT:float
VTK_PARSER_EXPONENT:int
VTK_PARSER_FLOOR:int
VTK_PARSER_GREATER_THAN:int
VTK_PARSER_HYPERBOLIC_COSINE:int
VTK_PARSER_HYPERBOLIC_SINE:int
VTK_PARSER_HYPERBOLIC_TANGENT:int
VTK_PARSER_IF:int
VTK_PARSER_IHAT:int
VTK_PARSER_IMMEDIATE:int
VTK_PARSER_JHAT:int
VTK_PARSER_KHAT:int
VTK_PARSER_LESS_THAN:int
VTK_PARSER_LOGARITHM:int
VTK_PARSER_LOGARITHM10:int
VTK_PARSER_LOGARITHME:int
VTK_PARSER_MAGNITUDE:int
VTK_PARSER_MAX:int
VTK_PARSER_MIN:int
VTK_PARSER_MULTIPLY:int
VTK_PARSER_NORMALIZE:int
VTK_PARSER_OR:int
VTK_PARSER_POWER:int
VTK_PARSER_SCALAR_TIMES_VECTOR:int
VTK_PARSER_SIGN:int
VTK_PARSER_SINE:int
VTK_PARSER_SQUARE_ROOT:int
VTK_PARSER_SUBTRACT:int
VTK_PARSER_TANGENT:int
VTK_PARSER_UNARY_MINUS:int
VTK_PARSER_UNARY_PLUS:int
VTK_PARSER_VECTOR_ADD:int
VTK_PARSER_VECTOR_IF:int
VTK_PARSER_VECTOR_OVER_SCALAR:int
VTK_PARSER_VECTOR_SUBTRACT:int
VTK_PARSER_VECTOR_TIMES_SCALAR:int
VTK_PARSER_VECTOR_UNARY_MINUS:int
VTK_PARSER_VECTOR_UNARY_PLUS:int

class vtkContourValues(vtkmodules.vtkCommonCore.vtkObject):
    def DeepCopy(self, other:'vtkContourValues') -> None: ...
    @overload
    def GenerateValues(self, numContours:int, range:MutableSequence[float]) -> None: ...
    @overload
    def GenerateValues(self, numContours:int, rangeStart:float, rangeEnd:float) -> None: ...
    def GetNumberOfContours(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetValue(self, i:int) -> float: ...
    @overload
    def GetValues(self) -> Pointer: ...
    @overload
    def GetValues(self, contourValues:MutableSequence[float]) -> None: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkContourValues': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkContourValues': ...
    def SetNumberOfContours(self, number:int) -> None: ...
    def SetValue(self, i:int, value:float) -> None: ...

class vtkErrorCode(object):
    class ErrorIds(int): ...
    CannotOpenFileError:'ErrorIds'
    FileFormatError:'ErrorIds'
    FileNotFoundError:'ErrorIds'
    FirstVTKErrorCode:'ErrorIds'
    NoError:'ErrorIds'
    NoFileNameError:'ErrorIds'
    OutOfDiskSpaceError:'ErrorIds'
    PrematureEndOfFileError:'ErrorIds'
    UnknownError:'ErrorIds'
    UnrecognizedFileTypeError:'ErrorIds'
    UserError:'ErrorIds'
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtkErrorCode') -> None: ...
    @staticmethod
    def GetErrorCodeFromString(error:str) -> int: ...
    @staticmethod
    def GetLastSystemError() -> int: ...
    @staticmethod
    def GetStringFromErrorCode(error:int) -> str: ...

class vtkExprTkFunctionParser(vtkmodules.vtkCommonCore.vtkObject):
    def GetFunction(self) -> str: ...
    def GetMTime(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfScalarVariables(self) -> int: ...
    def GetNumberOfVectorVariables(self) -> int: ...
    def GetReplaceInvalidValues(self) -> int: ...
    def GetReplacementValue(self) -> float: ...
    def GetScalarResult(self) -> float: ...
    def GetScalarVariableIndex(self, name:str) -> int: ...
    def GetScalarVariableName(self, i:int) -> str: ...
    @overload
    def GetScalarVariableNeeded(self, i:int) -> bool: ...
    @overload
    def GetScalarVariableNeeded(self, variableName:str) -> bool: ...
    @overload
    def GetScalarVariableValue(self, variableName:str) -> float: ...
    @overload
    def GetScalarVariableValue(self, i:int) -> float: ...
    @overload
    def GetVectorResult(self) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorResult(self, result:MutableSequence[float]) -> None: ...
    def GetVectorVariableIndex(self, name:str) -> int: ...
    def GetVectorVariableName(self, i:int) -> str: ...
    @overload
    def GetVectorVariableNeeded(self, i:int) -> bool: ...
    @overload
    def GetVectorVariableNeeded(self, variableName:str) -> bool: ...
    @overload
    def GetVectorVariableValue(self, variableName:str) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorVariableValue(self, variableName:str, value:MutableSequence[float]) -> None: ...
    @overload
    def GetVectorVariableValue(self, i:int) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorVariableValue(self, i:int, value:MutableSequence[float]) -> None: ...
    def InvalidateFunction(self) -> None: ...
    def IsA(self, type:str) -> int: ...
    def IsScalarResult(self) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def IsVectorResult(self) -> int: ...
    def NewInstance(self) -> 'vtkExprTkFunctionParser': ...
    def RemoveAllVariables(self) -> None: ...
    def RemoveScalarVariables(self) -> None: ...
    def RemoveVectorVariables(self) -> None: ...
    def ReplaceInvalidValuesOff(self) -> None: ...
    def ReplaceInvalidValuesOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkExprTkFunctionParser': ...
    @staticmethod
    def SanitizeName(name:str) -> str: ...
    def SetFunction(self, function:str) -> None: ...
    def SetReplaceInvalidValues(self, _arg:int) -> None: ...
    def SetReplacementValue(self, _arg:float) -> None: ...
    @overload
    def SetScalarVariableValue(self, variableName:str, value:float) -> None: ...
    @overload
    def SetScalarVariableValue(self, i:int, value:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, variableName:str, xValue:float, yValue:float, zValue:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, variableName:str, values:MutableSequence[float]) -> None: ...
    @overload
    def SetVectorVariableValue(self, i:int, xValue:float, yValue:float, zValue:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, i:int, values:MutableSequence[float]) -> None: ...

class vtkFunctionParser(vtkmodules.vtkCommonCore.vtkObject):
    def GetFunction(self) -> str: ...
    def GetMTime(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfScalarVariables(self) -> int: ...
    def GetNumberOfVectorVariables(self) -> int: ...
    def GetReplaceInvalidValues(self) -> int: ...
    def GetReplacementValue(self) -> float: ...
    def GetScalarResult(self) -> float: ...
    def GetScalarVariableIndex(self, name:str) -> int: ...
    def GetScalarVariableName(self, i:int) -> str: ...
    @overload
    def GetScalarVariableNeeded(self, i:int) -> bool: ...
    @overload
    def GetScalarVariableNeeded(self, variableName:str) -> bool: ...
    @overload
    def GetScalarVariableValue(self, variableName:str) -> float: ...
    @overload
    def GetScalarVariableValue(self, i:int) -> float: ...
    @overload
    def GetVectorResult(self) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorResult(self, result:MutableSequence[float]) -> None: ...
    def GetVectorVariableIndex(self, name:str) -> int: ...
    def GetVectorVariableName(self, i:int) -> str: ...
    @overload
    def GetVectorVariableNeeded(self, i:int) -> bool: ...
    @overload
    def GetVectorVariableNeeded(self, variableName:str) -> bool: ...
    @overload
    def GetVectorVariableValue(self, variableName:str) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorVariableValue(self, variableName:str, value:MutableSequence[float]) -> None: ...
    @overload
    def GetVectorVariableValue(self, i:int) -> Tuple[float, float, float]: ...
    @overload
    def GetVectorVariableValue(self, i:int, value:MutableSequence[float]) -> None: ...
    def InvalidateFunction(self) -> None: ...
    def IsA(self, type:str) -> int: ...
    def IsScalarResult(self) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def IsVectorResult(self) -> int: ...
    def NewInstance(self) -> 'vtkFunctionParser': ...
    def RemoveAllVariables(self) -> None: ...
    def RemoveScalarVariables(self) -> None: ...
    def RemoveVectorVariables(self) -> None: ...
    def ReplaceInvalidValuesOff(self) -> None: ...
    def ReplaceInvalidValuesOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkFunctionParser': ...
    def SetFunction(self, function:str) -> None: ...
    def SetReplaceInvalidValues(self, _arg:int) -> None: ...
    def SetReplacementValue(self, _arg:float) -> None: ...
    @overload
    def SetScalarVariableValue(self, variableName:str, value:float) -> None: ...
    @overload
    def SetScalarVariableValue(self, i:int, value:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, variableName:str, xValue:float, yValue:float, zValue:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, variableName:str, values:Sequence[float]) -> None: ...
    @overload
    def SetVectorVariableValue(self, i:int, xValue:float, yValue:float, zValue:float) -> None: ...
    @overload
    def SetVectorVariableValue(self, i:int, values:Sequence[float]) -> None: ...

class vtkHeap(vtkmodules.vtkCommonCore.vtkObject):
    def AllocateMemory(self, n:int) -> Pointer: ...
    def GetBlockSize(self) -> int: ...
    def GetNumberOfAllocations(self) -> int: ...
    def GetNumberOfBlocks(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkHeap': ...
    def Reset(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkHeap': ...
    def SetBlockSize(self, __a:int) -> None: ...
    def StringDup(self, str:str) -> str: ...

class vtkPolygonBuilder(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtkPolygonBuilder') -> None: ...
    def GetPolygons(self, polys:'vtkIdListCollection') -> None: ...
    def InsertTriangle(self, abc:Sequence[int]) -> None: ...
    def Reset(self) -> None: ...

class vtkResourceFileLocator(vtkmodules.vtkCommonCore.vtkObject):
    @staticmethod
    def GetLibraryPathForSymbolUnix(symbolname:str) -> str: ...
    @staticmethod
    def GetLibraryPathForSymbolWin32(fptr:Pointer) -> str: ...
    def GetLogVerbosity(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    @overload
    def Locate(self, anchor:str, landmark:str, defaultDir:str=...) -> str: ...
    @overload
    def Locate(self, anchor:str, landmark_prefixes:Sequence[str], landmark:str, defaultDir:str=...) -> str: ...
    def NewInstance(self) -> 'vtkResourceFileLocator': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkResourceFileLocator': ...
    def SetLogVerbosity(self, _arg:int) -> None: ...
