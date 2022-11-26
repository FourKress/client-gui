from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore
import vtkmodules.vtkCommonExecutionModel
import vtkmodules.vtkIOCore

VTK_SQL_ALLBACKENDS:str
VTK_SQL_DEFAULT_COLUMN_SIZE:int
VTK_SQL_FEATURE_BATCH_OPERATIONS:int
VTK_SQL_FEATURE_BLOB:int
VTK_SQL_FEATURE_LAST_INSERT_ID:int
VTK_SQL_FEATURE_NAMED_PLACEHOLDERS:int
VTK_SQL_FEATURE_POSITIONAL_PLACEHOLDERS:int
VTK_SQL_FEATURE_PREPARED_QUERIES:int
VTK_SQL_FEATURE_QUERY_SIZE:int
VTK_SQL_FEATURE_TRANSACTIONS:int
VTK_SQL_FEATURE_TRIGGERS:int
VTK_SQL_FEATURE_UNICODE:int
VTK_SQL_MYSQL:str
VTK_SQL_POSTGRESQL:str
VTK_SQL_SQLITE:str

class vtkDatabaseToTableReader(vtkmodules.vtkCommonExecutionModel.vtkTableAlgorithm):
    def CheckIfTableExists(self) -> bool: ...
    def GetDatabase(self) -> 'vtkSQLDatabase': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkDatabaseToTableReader': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkDatabaseToTableReader': ...
    def SetDatabase(self, db:'vtkSQLDatabase') -> bool: ...
    def SetTableName(self, name:str) -> bool: ...

class vtkRowQuery(vtkmodules.vtkCommonCore.vtkObject):
    def CaseSensitiveFieldNamesOff(self) -> None: ...
    def CaseSensitiveFieldNamesOn(self) -> None: ...
    def DataValue(self, c:int) -> 'vtkVariant': ...
    def Execute(self) -> bool: ...
    def GetCaseSensitiveFieldNames(self) -> bool: ...
    def GetFieldIndex(self, name:str) -> int: ...
    def GetFieldName(self, i:int) -> str: ...
    def GetFieldType(self, i:int) -> int: ...
    def GetLastErrorText(self) -> str: ...
    def GetNumberOfFields(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def HasError(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    def IsActive(self) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkRowQuery': ...
    @overload
    def NextRow(self) -> bool: ...
    @overload
    def NextRow(self, rowArray:'vtkVariantArray') -> bool: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkRowQuery': ...
    def SetCaseSensitiveFieldNames(self, _arg:bool) -> None: ...

class vtkRowQueryToTable(vtkmodules.vtkCommonExecutionModel.vtkTableAlgorithm):
    def GetMTime(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetQuery(self) -> 'vtkRowQuery': ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkRowQueryToTable': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkRowQueryToTable': ...
    def SetQuery(self, query:'vtkRowQuery') -> None: ...

class vtkSQLDatabase(vtkmodules.vtkCommonCore.vtkObject):
    def Close(self) -> None: ...
    @staticmethod
    def CreateFromURL(URL:str) -> 'vtkSQLDatabase': ...
    @staticmethod
    def DATABASE() -> 'vtkInformationObjectBaseKey': ...
    def EffectSchema(self, __a:'vtkSQLDatabaseSchema', dropIfExists:bool=False) -> bool: ...
    def GetColumnSpecification(self, schema:'vtkSQLDatabaseSchema', tblHandle:int, colHandle:int) -> str: ...
    def GetDatabaseType(self) -> str: ...
    def GetIndexSpecification(self, schema:'vtkSQLDatabaseSchema', tblHandle:int, idxHandle:int, skipped:bool) -> str: ...
    def GetLastErrorText(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetQueryInstance(self) -> 'vtkSQLQuery': ...
    def GetRecord(self, table:str) -> 'vtkStringArray': ...
    def GetTablePreamble(self, __a:bool) -> str: ...
    def GetTables(self) -> 'vtkStringArray': ...
    def GetTriggerSpecification(self, schema:'vtkSQLDatabaseSchema', tblHandle:int, trgHandle:int) -> str: ...
    def GetURL(self) -> str: ...
    def HasError(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    def IsOpen(self) -> bool: ...
    def IsSupported(self, feature:int) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLDatabase': ...
    def Open(self, password:str) -> bool: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLDatabase': ...
    @staticmethod
    def UnRegisterAllCreateFromURLCallbacks() -> None: ...

class vtkSQLDatabaseSchema(vtkmodules.vtkCommonCore.vtkObject):
    class DatabaseTriggerType(int): ...
    class DatabaseColumnType(int): ...
    class VarargTokens(int): ...
    class DatabaseIndexType(int): ...
    AFTER_DELETE:'DatabaseTriggerType'
    AFTER_INSERT:'DatabaseTriggerType'
    AFTER_UPDATE:'DatabaseTriggerType'
    BEFORE_DELETE:'DatabaseTriggerType'
    BEFORE_INSERT:'DatabaseTriggerType'
    BEFORE_UPDATE:'DatabaseTriggerType'
    BIGINT:'DatabaseColumnType'
    BLOB:'DatabaseColumnType'
    COLUMN_TOKEN:'VarargTokens'
    DATE:'DatabaseColumnType'
    DOUBLE:'DatabaseColumnType'
    END_INDEX_TOKEN:'VarargTokens'
    END_TABLE_TOKEN:'VarargTokens'
    INDEX:'DatabaseIndexType'
    INDEX_COLUMN_TOKEN:'VarargTokens'
    INDEX_TOKEN:'VarargTokens'
    INTEGER:'DatabaseColumnType'
    OPTION_TOKEN:'VarargTokens'
    PRIMARY_KEY:'DatabaseIndexType'
    REAL:'DatabaseColumnType'
    SERIAL:'DatabaseColumnType'
    SMALLINT:'DatabaseColumnType'
    TEXT:'DatabaseColumnType'
    TIME:'DatabaseColumnType'
    TIMESTAMP:'DatabaseColumnType'
    TRIGGER_TOKEN:'VarargTokens'
    UNIQUE:'DatabaseIndexType'
    VARCHAR:'DatabaseColumnType'
    @overload
    def AddColumnToIndex(self, tblHandle:int, idxHandle:int, colHandle:int) -> int: ...
    @overload
    def AddColumnToIndex(self, tblName:str, idxName:str, colName:str) -> int: ...
    @overload
    def AddColumnToTable(self, tblHandle:int, colType:int, colName:str, colSize:int, colOpts:str) -> int: ...
    @overload
    def AddColumnToTable(self, tblName:str, colType:int, colName:str, colSize:int, colAttribs:str) -> int: ...
    @overload
    def AddIndexToTable(self, tblHandle:int, idxType:int, idxName:str) -> int: ...
    @overload
    def AddIndexToTable(self, tblName:str, idxType:int, idxName:str) -> int: ...
    @overload
    def AddOptionToTable(self, tblHandle:int, optText:str, optBackend:str=...) -> int: ...
    @overload
    def AddOptionToTable(self, tblName:str, optStr:str, optBackend:str=...) -> int: ...
    def AddPreamble(self, preName:str, preAction:str, preBackend:str=...) -> int: ...
    def AddTable(self, tblName:str) -> int: ...
    def AddTableMultipleArguments(self, tblName:str) -> int: ...
    @overload
    def AddTriggerToTable(self, tblHandle:int, trgType:int, trgName:str, trgAction:str, trgBackend:str=...) -> int: ...
    @overload
    def AddTriggerToTable(self, tblName:str, trgType:int, trgName:str, trgAction:str, trgBackend:str=...) -> int: ...
    def GetColumnAttributesFromHandle(self, tblHandle:int, colHandle:int) -> str: ...
    def GetColumnHandleFromName(self, tblName:str, colName:str) -> int: ...
    def GetColumnNameFromHandle(self, tblHandle:int, colHandle:int) -> str: ...
    def GetColumnSizeFromHandle(self, tblHandle:int, colHandle:int) -> int: ...
    def GetColumnTypeFromHandle(self, tblHandle:int, colHandle:int) -> int: ...
    def GetIndexColumnNameFromHandle(self, tblHandle:int, idxHandle:int, cnmHandle:int) -> str: ...
    def GetIndexHandleFromName(self, tblName:str, idxName:str) -> int: ...
    def GetIndexNameFromHandle(self, tblHandle:int, idxHandle:int) -> str: ...
    def GetIndexTypeFromHandle(self, tblHandle:int, idxHandle:int) -> int: ...
    def GetName(self) -> str: ...
    def GetNumberOfColumnNamesInIndex(self, tblHandle:int, idxHandle:int) -> int: ...
    def GetNumberOfColumnsInTable(self, tblHandle:int) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfIndicesInTable(self, tblHandle:int) -> int: ...
    def GetNumberOfOptionsInTable(self, tblHandle:int) -> int: ...
    def GetNumberOfPreambles(self) -> int: ...
    def GetNumberOfTables(self) -> int: ...
    def GetNumberOfTriggersInTable(self, tblHandle:int) -> int: ...
    def GetOptionBackendFromHandle(self, tblHandle:int, optHandle:int) -> str: ...
    def GetOptionTextFromHandle(self, tblHandle:int, optHandle:int) -> str: ...
    def GetPreambleActionFromHandle(self, preHandle:int) -> str: ...
    def GetPreambleBackendFromHandle(self, preHandle:int) -> str: ...
    def GetPreambleHandleFromName(self, preName:str) -> int: ...
    def GetPreambleNameFromHandle(self, preHandle:int) -> str: ...
    def GetTableHandleFromName(self, tblName:str) -> int: ...
    def GetTableNameFromHandle(self, tblHandle:int) -> str: ...
    def GetTriggerActionFromHandle(self, tblHandle:int, trgHandle:int) -> str: ...
    def GetTriggerBackendFromHandle(self, tblHandle:int, trgHandle:int) -> str: ...
    def GetTriggerHandleFromName(self, tblName:str, trgName:str) -> int: ...
    def GetTriggerNameFromHandle(self, tblHandle:int, trgHandle:int) -> str: ...
    def GetTriggerTypeFromHandle(self, tblHandle:int, trgHandle:int) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLDatabaseSchema': ...
    def Reset(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLDatabaseSchema': ...
    def SetName(self, _arg:str) -> None: ...

class vtkSQLDatabaseTableSource(vtkmodules.vtkCommonExecutionModel.vtkTableAlgorithm):
    def GeneratePedigreeIdsOff(self) -> None: ...
    def GeneratePedigreeIdsOn(self) -> None: ...
    def GetGeneratePedigreeIds(self) -> bool: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetPedigreeIdArrayName(self) -> str: ...
    def GetQuery(self) -> str: ...
    def GetURL(self) -> str: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLDatabaseTableSource': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLDatabaseTableSource': ...
    def SetGeneratePedigreeIds(self, _arg:bool) -> None: ...
    def SetPassword(self, password:str) -> None: ...
    def SetPedigreeIdArrayName(self, _arg:str) -> None: ...
    def SetQuery(self, query:str) -> None: ...
    def SetURL(self, url:str) -> None: ...

class vtkSQLQuery(vtkRowQuery):
    def BeginTransaction(self) -> bool: ...
    @overload
    def BindParameter(self, index:int, value:int) -> bool: ...
    @overload
    def BindParameter(self, index:int, value:float) -> bool: ...
    @overload
    def BindParameter(self, index:int, stringValue:str, length:int) -> bool: ...
    @overload
    def BindParameter(self, index:int, string:str) -> bool: ...
    @overload
    def BindParameter(self, index:int, var:'vtkVariant') -> bool: ...
    @overload
    def BindParameter(self, index:int, data:Pointer, length:int) -> bool: ...
    def ClearParameterBindings(self) -> bool: ...
    def CommitTransaction(self) -> bool: ...
    def EscapeString(self, s:str, addSurroundingQuotes:bool=True) -> str: ...
    def Execute(self) -> bool: ...
    def GetDatabase(self) -> 'vtkSQLDatabase': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetQuery(self) -> str: ...
    def IsA(self, type:str) -> int: ...
    def IsActive(self) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLQuery': ...
    def RollbackTransaction(self) -> bool: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLQuery': ...
    def SetQuery(self, query:str) -> bool: ...

class vtkSQLiteDatabase(vtkSQLDatabase):
    CREATE:int
    CREATE_OR_CLEAR:int
    USE_EXISTING:int
    USE_EXISTING_OR_CREATE:int
    def Close(self) -> None: ...
    def GetColumnSpecification(self, schema:'vtkSQLDatabaseSchema', tblHandle:int, colHandle:int) -> str: ...
    def GetDatabaseFileName(self) -> str: ...
    def GetDatabaseType(self) -> str: ...
    def GetLastErrorText(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetQueryInstance(self) -> 'vtkSQLQuery': ...
    def GetRecord(self, table:str) -> 'vtkStringArray': ...
    def GetTables(self) -> 'vtkStringArray': ...
    def GetURL(self) -> str: ...
    def HasError(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    def IsOpen(self) -> bool: ...
    def IsSupported(self, feature:int) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLiteDatabase': ...
    @overload
    def Open(self, password:str) -> bool: ...
    @overload
    def Open(self, password:str, mode:int) -> bool: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLiteDatabase': ...
    def SetDatabaseFileName(self, _arg:str) -> None: ...

class vtkSQLiteQuery(vtkSQLQuery):
    def BeginTransaction(self) -> bool: ...
    @overload
    def BindParameter(self, index:int, value:int) -> bool: ...
    @overload
    def BindParameter(self, index:int, value:float) -> bool: ...
    @overload
    def BindParameter(self, index:int, stringValue:str, length:int) -> bool: ...
    @overload
    def BindParameter(self, index:int, string:str) -> bool: ...
    @overload
    def BindParameter(self, index:int, value:'vtkVariant') -> bool: ...
    @overload
    def BindParameter(self, index:int, data:Pointer, length:int) -> bool: ...
    def ClearParameterBindings(self) -> bool: ...
    def CommitTransaction(self) -> bool: ...
    def DataValue(self, c:int) -> 'vtkVariant': ...
    def Execute(self) -> bool: ...
    def GetFieldName(self, i:int) -> str: ...
    def GetFieldType(self, i:int) -> int: ...
    def GetLastErrorText(self) -> str: ...
    def GetNumberOfFields(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def HasError(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLiteQuery': ...
    def NextRow(self) -> bool: ...
    def RollbackTransaction(self) -> bool: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLiteQuery': ...
    def SetQuery(self, query:str) -> bool: ...

class vtkSQLiteToTableReader(vtkDatabaseToTableReader):
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSQLiteToTableReader': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSQLiteToTableReader': ...

class vtkTableToDatabaseWriter(vtkmodules.vtkIOCore.vtkWriter):
    def GetDatabase(self) -> 'vtkSQLDatabase': ...
    @overload
    def GetInput(self) -> 'vtkTable': ...
    @overload
    def GetInput(self, port:int) -> 'vtkTable': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkTableToDatabaseWriter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkTableToDatabaseWriter': ...
    def SetDatabase(self, db:'vtkSQLDatabase') -> bool: ...
    def SetTableName(self, name:str) -> bool: ...
    def TableNameIsNew(self) -> bool: ...

class vtkTableToSQLiteWriter(vtkTableToDatabaseWriter):
    @overload
    def GetInput(self) -> 'vtkTable': ...
    @overload
    def GetInput(self, port:int) -> 'vtkTable': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkTableToSQLiteWriter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkTableToSQLiteWriter': ...

