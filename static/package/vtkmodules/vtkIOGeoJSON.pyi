from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore
import vtkmodules.vtkCommonDataModel
import vtkmodules.vtkCommonExecutionModel
import vtkmodules.vtkIOCore

class vtkGeoJSONFeature(vtkmodules.vtkCommonDataModel.vtkDataObject):
    def GetDataObjectType(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutlinePolygons(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkGeoJSONFeature': ...
    def OutlinePolygonsOff(self) -> None: ...
    def OutlinePolygonsOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkGeoJSONFeature': ...
    def SetOutlinePolygons(self, _arg:bool) -> None: ...

class vtkGeoJSONReader(vtkmodules.vtkCommonExecutionModel.vtkPolyDataAlgorithm):
    def AddFeatureProperty(self, name:str, typeAndDefaultValue:'vtkVariant') -> None: ...
    def GetFileName(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutlinePolygons(self) -> bool: ...
    def GetSerializedPropertiesArrayName(self) -> str: ...
    def GetStringInput(self) -> str: ...
    def GetStringInputMode(self) -> bool: ...
    def GetTriangulatePolygons(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkGeoJSONReader': ...
    def OutlinePolygonsOff(self) -> None: ...
    def OutlinePolygonsOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkGeoJSONReader': ...
    def SetFileName(self, _arg:str) -> None: ...
    def SetOutlinePolygons(self, _arg:bool) -> None: ...
    def SetSerializedPropertiesArrayName(self, _arg:str) -> None: ...
    def SetStringInput(self, _arg:str) -> None: ...
    def SetStringInputMode(self, _arg:bool) -> None: ...
    def SetTriangulatePolygons(self, _arg:bool) -> None: ...
    def StringInputModeOff(self) -> None: ...
    def StringInputModeOn(self) -> None: ...
    def TriangulatePolygonsOff(self) -> None: ...
    def TriangulatePolygonsOn(self) -> None: ...

class vtkGeoJSONWriter(vtkmodules.vtkIOCore.vtkWriter):
    def GetBinaryOutputString(self) -> Pointer: ...
    def GetFileName(self) -> str: ...
    def GetLookupTable(self) -> 'vtkLookupTable': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputStdString(self) -> str: ...
    def GetOutputString(self) -> str: ...
    def GetOutputStringLength(self) -> int: ...
    def GetScalarFormat(self) -> int: ...
    def GetWriteToOutputString(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkGeoJSONWriter': ...
    def RegisterAndGetOutputString(self) -> str: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkGeoJSONWriter': ...
    def SetFileName(self, _arg:str) -> None: ...
    def SetLookupTable(self, lut:'vtkLookupTable') -> None: ...
    def SetScalarFormat(self, _arg:int) -> None: ...
    def SetWriteToOutputString(self, _arg:bool) -> None: ...
    def WriteToOutputStringOff(self) -> None: ...
    def WriteToOutputStringOn(self) -> None: ...

