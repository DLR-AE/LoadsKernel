# -*- coding: utf-8 -*-
#############################################################################
# Copyright (C) 2007-2013 German Aerospace Center (DLR/SC)
#
# Created: 2013-05-01 Martin Siggel <Martin.Siggel@dlr.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################

# This file is automatically created from tixi.h on 2015-12-11.
# If you experience any bugs please contact the authors

import sys, ctypes

class StorageMode(object):
    ROW_WISE = 0
    COLUMN_WISE = 1
    _names = {}
    _names[0] = 'ROW_WISE'
    _names[1] = 'COLUMN_WISE'


class ReturnCode(object):
    SUCCESS = 0
    FAILED = 1
    INVALID_XML_NAME = 2
    NOT_WELL_FORMED = 3
    NOT_SCHEMA_COMPLIANT = 4
    NOT_DTD_COMPLIANT = 5
    INVALID_HANDLE = 6
    INVALID_XPATH = 7
    ELEMENT_NOT_FOUND = 8
    INDEX_OUT_OF_RANGE = 9
    NO_POINT_FOUND = 10
    NOT_AN_ELEMENT = 11
    ATTRIBUTE_NOT_FOUND = 12
    OPEN_FAILED = 13
    OPEN_SCHEMA_FAILED = 14
    OPEN_DTD_FAILED = 15
    CLOSE_FAILED = 16
    ALREADY_SAVED = 17
    ELEMENT_PATH_NOT_UNIQUE = 18
    NO_ELEMENT_NAME = 19
    NO_CHILDREN = 20
    CHILD_NOT_FOUND = 21
    EROROR_CREATE_ROOT_NODE = 22
    DEALLOCATION_FAILED = 23
    NO_NUMBER = 24
    NO_ATTRIBUTE_NAME = 25
    STRING_TRUNCATED = 26
    NON_MATCHING_NAME = 27
    NON_MATCHING_SIZE = 28
    MATRIX_DIMENSION_ERROR = 29
    COORDINATE_NOT_FOUND = 30
    UNKNOWN_STORAGE_MODE = 31
    UID_NOT_UNIQUE = 32
    UID_DONT_EXISTS = 33
    UID_LINK_BROKEN = 34
    _names = {}
    _names[0] = 'SUCCESS'
    _names[1] = 'FAILED'
    _names[2] = 'INVALID_XML_NAME'
    _names[3] = 'NOT_WELL_FORMED'
    _names[4] = 'NOT_SCHEMA_COMPLIANT'
    _names[5] = 'NOT_DTD_COMPLIANT'
    _names[6] = 'INVALID_HANDLE'
    _names[7] = 'INVALID_XPATH'
    _names[8] = 'ELEMENT_NOT_FOUND'
    _names[9] = 'INDEX_OUT_OF_RANGE'
    _names[10] = 'NO_POINT_FOUND'
    _names[11] = 'NOT_AN_ELEMENT'
    _names[12] = 'ATTRIBUTE_NOT_FOUND'
    _names[13] = 'OPEN_FAILED'
    _names[14] = 'OPEN_SCHEMA_FAILED'
    _names[15] = 'OPEN_DTD_FAILED'
    _names[16] = 'CLOSE_FAILED'
    _names[17] = 'ALREADY_SAVED'
    _names[18] = 'ELEMENT_PATH_NOT_UNIQUE'
    _names[19] = 'NO_ELEMENT_NAME'
    _names[20] = 'NO_CHILDREN'
    _names[21] = 'CHILD_NOT_FOUND'
    _names[22] = 'EROROR_CREATE_ROOT_NODE'
    _names[23] = 'DEALLOCATION_FAILED'
    _names[24] = 'NO_NUMBER'
    _names[25] = 'NO_ATTRIBUTE_NAME'
    _names[26] = 'STRING_TRUNCATED'
    _names[27] = 'NON_MATCHING_NAME'
    _names[28] = 'NON_MATCHING_SIZE'
    _names[29] = 'MATRIX_DIMENSION_ERROR'
    _names[30] = 'COORDINATE_NOT_FOUND'
    _names[31] = 'UNKNOWN_STORAGE_MODE'
    _names[32] = 'UID_NOT_UNIQUE'
    _names[33] = 'UID_DONT_EXISTS'
    _names[34] = 'UID_LINK_BROKEN'


class MessageType(object):
    MESSAGETYPE_ERROR = 0
    MESSAGETYPE_WARNING = 1
    MESSAGETYPE_STATUS = 2
    _names = {}
    _names[0] = 'MESSAGETYPE_ERROR'
    _names[1] = 'MESSAGETYPE_WARNING'
    _names[2] = 'MESSAGETYPE_STATUS'


class OpenMode(object):
    OPENMODE_PLAIN = 0
    OPENMODE_RECURSIVE = 1
    _names = {}
    _names[0] = 'OPENMODE_PLAIN'
    _names[1] = 'OPENMODE_RECURSIVE'


class TixiException(Exception):
    ''' The exception encapsulates the error return code of the library and arguments that were provided for the function. '''
    def __init__(self, code, *args, **kwargs):
        Exception.__init__(self)
        self.code = code
        if "error" in kwargs:
            self.error = str(kwargs["error"])
        elif code in ReturnCode._names:
            self.error = ReturnCode._names[code]
        else:
            self.error = "UNDEFINED"
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
    def __str__(self):
        return self.error + " (" + str(self.code) + ") " + str(list(self.args)) + " " + str(self.kwargs)


def catch_error(returncode, *args, **kwargs):
    if returncode != ReturnCode.SUCCESS:
        raise TixiException(returncode, args, kwargs)



def encode_for_c(thestring):
    if type(thestring) is str:
        return str.encode(thestring)
    else:
        return thestring
        
def decode_for_py(thestring):
    if sys.version_info[0] >= 3:
        return thestring.decode("utf-8")
    else:
        return thestring


class Tixi(object):

    # load the TIXI library
    # We only support python2.5 and newer
    if sys.version_info<(2,5,0):
        print("At least python 2.5 is needed from tixiWrapper.")

    try:
        if sys.platform == 'win32':
            lib = ctypes.cdll.TIXI
        elif sys.platform == 'darwin':
            lib = ctypes.CDLL("libTIXI.dylib")
        else:
            lib = ctypes.CDLL("libTIXI.so")
    except:
        raise Exception("Could not load the TIXI library. Please check if:\n" +
        "  1) The PATH (Windows) / LD_LIBRARY_PATH (Linux) environment variable points to the library\n" +
        "  2) The architecture of the library matches the architecture of python (a 32 bit python needs a 32 bit shared library)\n")

    def __init__(self):
        self._handle = ctypes.c_int(-1)

        
        self.version = self.getVersion()


    def __del__(self):
        if hasattr(self, "lib"):
            if self.lib != None:
                self.close()
                self.lib = None


    def open(self, xmlInputFilename, recursive = False):
        if recursive:
            self.openDocumentRecursive(xmlInputFilename, OpenMode.OPENMODE_RECURSIVE)
        else:
            self.openDocument(xmlInputFilename)
    
    def close(self):
        if self._handle.value >= 0:
            ret = self.lib.tixiCloseDocument(self._handle)
            self._handle.value = -1
            catch_error(ret)
    
    def save(self, fileName, recursive = False, remove = False):
        ''' Save the main tixi document.
            If the document was opened recursively,
             * 'recursive' tells to save linked nodes to their respecitve files, too.
             * 'remove' tells to remove the links to external files after saving the complete CPACS inclusively all linked content to the main file.
            You cannot have 'remove' without 'recursive'.
        '''
        if recursive and remove:
            self.saveAndRemoveDocument(fileName)
        elif recursive:
            self.saveCompleteDocument(fileName)
        else:
            self.saveDocument(fileName)
    
    def checkElement(self, elementPath):
        ''' boolean return values from special return code is coded manually here '''
        _c_elementPath = ctypes.c_char_p()
        _c_elementPath.value = str.encode(elementPath)
        tixiReturn = self.lib.tixiCheckElement(self._handle, _c_elementPath)
        if tixiReturn == ReturnCode.SUCCESS:
            return True
        if tixiReturn == ReturnCode.ELEMENT_NOT_FOUND:
            return False
        catch_error(tixiReturn, elementPath)
        
    def uIDCheckExists(self, uID):
        _c_uID = ctypes.c_char_p()
        _c_uID.value = str.encode(uID)
        tixiReturn = self.lib.tixiUIDCheckExists(self._handle, _c_uID)
        if tixiReturn == ReturnCode.SUCCESS:
            return True
        else:
            return False
        catch_error(tixiReturn, uID) 
        
    def checkAttribute(self, elementPath, attributeName):
        ''' boolean return values from special return code is coded manually here '''
        _c_elementPath = ctypes.c_char_p()
        _c_elementPath.value = str.encode(elementPath)
        _c_attributeName = ctypes.c_char_p()
        _c_attributeName.value = str.encode(attributeName)
        tixiReturn = self.lib.tixiCheckAttribute(self._handle, _c_elementPath, _c_attributeName)
        if tixiReturn == ReturnCode.SUCCESS:
            return True
        if tixiReturn == ReturnCode.ATTRIBUTE_NOT_FOUND:
            return False
        catch_error(tixiReturn, elementPath, attributeName)
        
        
    def getVersion(self):

        # call to native function
        self.lib.tixiGetVersion.restype = ctypes.c_char_p
        _c_ret = self.lib.tixiGetVersion()

        _py_ret = decode_for_py(_c_ret)

        return _py_ret



    def openDocument(self, xmlFilename):
        # input arg conversion
        _c_xmlFilename = ctypes.c_char_p(encode_for_c(xmlFilename))

        # call to native function
        errorCode = self.lib.tixiOpenDocument(_c_xmlFilename, ctypes.byref(self._handle))
        catch_error(errorCode, 'tixiOpenDocument', xmlFilename)




    def openDocumentRecursive(self, xmlFilename, oMode):
        # input arg conversion
        _c_xmlFilename = ctypes.c_char_p(encode_for_c(xmlFilename))
        _c_oMode = ctypes.c_int(oMode)

        # call to native function
        errorCode = self.lib.tixiOpenDocumentRecursive(_c_xmlFilename, ctypes.byref(self._handle), _c_oMode)
        catch_error(errorCode, 'tixiOpenDocumentRecursive', xmlFilename, oMode)




    def openHttp(self, httpURL):
        # input arg conversion
        _c_httpURL = ctypes.c_char_p(encode_for_c(httpURL))

        # call to native function
        errorCode = self.lib.tixiOpenDocumentFromHTTP(_c_httpURL, ctypes.byref(self._handle))
        catch_error(errorCode, 'tixiOpenDocumentFromHTTP', httpURL)




    def create(self, rootElementName):
        # input arg conversion
        _c_rootElementName = ctypes.c_char_p(encode_for_c(rootElementName))

        # call to native function
        errorCode = self.lib.tixiCreateDocument(_c_rootElementName, ctypes.byref(self._handle))
        catch_error(errorCode, 'tixiCreateDocument', rootElementName)




    def getDocumentPath(self):

        # output arg preparation
        _c_documentPath = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetDocumentPath(self._handle, ctypes.byref(_c_documentPath))
        catch_error(errorCode, 'tixiGetDocumentPath')

        _py_documentPath = decode_for_py(_c_documentPath.value)

        return _py_documentPath



    def saveDocument(self, xmlFilename):
        # input arg conversion
        _c_xmlFilename = ctypes.c_char_p(encode_for_c(xmlFilename))

        # call to native function
        errorCode = self.lib.tixiSaveDocument(self._handle, _c_xmlFilename)
        catch_error(errorCode, 'tixiSaveDocument', xmlFilename)



    def saveCompleteDocument(self, xmlFilename):
        # input arg conversion
        _c_xmlFilename = ctypes.c_char_p(encode_for_c(xmlFilename))

        # call to native function
        errorCode = self.lib.tixiSaveCompleteDocument(self._handle, _c_xmlFilename)
        catch_error(errorCode, 'tixiSaveCompleteDocument', xmlFilename)



    def saveAndRemoveDocument(self, xmlFilename):
        # input arg conversion
        _c_xmlFilename = ctypes.c_char_p(encode_for_c(xmlFilename))

        # call to native function
        errorCode = self.lib.tixiSaveAndRemoveDocument(self._handle, _c_xmlFilename)
        catch_error(errorCode, 'tixiSaveAndRemoveDocument', xmlFilename)



    def closeAllDocuments(self):

        # call to native function
        errorCode = self.lib.tixiCloseAllDocuments()
        catch_error(errorCode, 'tixiCloseAllDocuments')



    def cleanup(self):

        # call to native function
        errorCode = self.lib.tixiCleanup()
        catch_error(errorCode, 'tixiCleanup')



    def exportDocumentAsString(self):

        # output arg preparation
        _c_text = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiExportDocumentAsString(self._handle, ctypes.byref(_c_text))
        catch_error(errorCode, 'tixiExportDocumentAsString')

        _py_text = decode_for_py(_c_text.value)

        return _py_text



    def openString(self, xmlImportString):
        # input arg conversion
        _c_xmlImportString = ctypes.c_char_p(encode_for_c(xmlImportString))

        # call to native function
        errorCode = self.lib.tixiImportFromString(_c_xmlImportString, ctypes.byref(self._handle))
        catch_error(errorCode, 'tixiImportFromString', xmlImportString)




    def schemaValidateFromFile(self, xsdFilename):
        # input arg conversion
        _c_xsdFilename = ctypes.c_char_p(encode_for_c(xsdFilename))

        # call to native function
        errorCode = self.lib.tixiSchemaValidateFromFile(self._handle, _c_xsdFilename)
        catch_error(errorCode, 'tixiSchemaValidateFromFile', xsdFilename)



    def schemaValidateFromString(self, xsdString):
        # input arg conversion
        _c_xsdString = ctypes.c_char_p(encode_for_c(xsdString))

        # call to native function
        errorCode = self.lib.tixiSchemaValidateFromString(self._handle, _c_xsdString)
        catch_error(errorCode, 'tixiSchemaValidateFromString', xsdString)



    def dTDValidate(self, DTDFilename):
        # input arg conversion
        _c_DTDFilename = ctypes.c_char_p(encode_for_c(DTDFilename))

        # call to native function
        errorCode = self.lib.tixiDTDValidate(self._handle, _c_DTDFilename)
        catch_error(errorCode, 'tixiDTDValidate', DTDFilename)



    def getTextElement(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_text = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetTextElement(self._handle, _c_elementPath, ctypes.byref(_c_text))
        catch_error(errorCode, 'tixiGetTextElement', elementPath)

        _py_text = decode_for_py(_c_text.value)

        return _py_text



    def getIntegerElement(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_number = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetIntegerElement(self._handle, _c_elementPath, ctypes.byref(_c_number))
        catch_error(errorCode, 'tixiGetIntegerElement', elementPath)

        _py_number = _c_number.value

        return _py_number



    def getDoubleElement(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_number = ctypes.c_double()

        # call to native function
        errorCode = self.lib.tixiGetDoubleElement(self._handle, _c_elementPath, ctypes.byref(_c_number))
        catch_error(errorCode, 'tixiGetDoubleElement', elementPath)

        _py_number = _c_number.value

        return _py_number



    def getBooleanElement(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_boolean = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetBooleanElement(self._handle, _c_elementPath, ctypes.byref(_c_boolean))
        catch_error(errorCode, 'tixiGetBooleanElement', elementPath)

        _py_boolean = _c_boolean.value

        return _py_boolean



    def updateTextElement(self, elementPath, text):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_text = ctypes.c_char_p(encode_for_c(text))

        # call to native function
        errorCode = self.lib.tixiUpdateTextElement(self._handle, _c_elementPath, _c_text)
        catch_error(errorCode, 'tixiUpdateTextElement', elementPath, text)



    def updateDoubleElement(self, elementPath, number, format):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_number = ctypes.c_double(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiUpdateDoubleElement(self._handle, _c_elementPath, _c_number, _c_format)
        catch_error(errorCode, 'tixiUpdateDoubleElement', elementPath, number, format)



    def updateIntegerElement(self, elementPath, number, format):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_number = ctypes.c_int(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiUpdateIntegerElement(self._handle, _c_elementPath, _c_number, _c_format)
        catch_error(errorCode, 'tixiUpdateIntegerElement', elementPath, number, format)



    def updateBooleanElement(self, elementPath, boolean):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_boolean = ctypes.c_int(boolean)

        # call to native function
        errorCode = self.lib.tixiUpdateBooleanElement(self._handle, _c_elementPath, _c_boolean)
        catch_error(errorCode, 'tixiUpdateBooleanElement', elementPath, boolean)



    def addTextElement(self, parentPath, elementName, text):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_text = ctypes.c_char_p(encode_for_c(text))

        # call to native function
        errorCode = self.lib.tixiAddTextElement(self._handle, _c_parentPath, _c_elementName, _c_text)
        catch_error(errorCode, 'tixiAddTextElement', parentPath, elementName, text)



    def addTextElementAtIndex(self, parentPath, elementName, text, index):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_text = ctypes.c_char_p(encode_for_c(text))
        _c_index = ctypes.c_int(index)

        # call to native function
        errorCode = self.lib.tixiAddTextElementAtIndex(self._handle, _c_parentPath, _c_elementName, _c_text, _c_index)
        catch_error(errorCode, 'tixiAddTextElementAtIndex', parentPath, elementName, text, index)



    def addBooleanElement(self, parentPath, elementName, boolean):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_boolean = ctypes.c_int(boolean)

        # call to native function
        errorCode = self.lib.tixiAddBooleanElement(self._handle, _c_parentPath, _c_elementName, _c_boolean)
        catch_error(errorCode, 'tixiAddBooleanElement', parentPath, elementName, boolean)



    def addDoubleElement(self, parentPath, elementName, number, format):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_number = ctypes.c_double(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddDoubleElement(self._handle, _c_parentPath, _c_elementName, _c_number, _c_format)
        catch_error(errorCode, 'tixiAddDoubleElement', parentPath, elementName, number, format)



    def addIntegerElement(self, parentPath, elementName, number, format):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_number = ctypes.c_int(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddIntegerElement(self._handle, _c_parentPath, _c_elementName, _c_number, _c_format)
        catch_error(errorCode, 'tixiAddIntegerElement', parentPath, elementName, number, format)



    def addFloatVector(self, parentPath, elementName, vector, numElements, format):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        array_t_vector = ctypes.c_double * len(vector)
        _c_vector = array_t_vector(*vector)
        _c_numElements = ctypes.c_int(numElements)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddFloatVector(self._handle, _c_parentPath, _c_elementName, _c_vector, _c_numElements, _c_format)
        catch_error(errorCode, 'tixiAddFloatVector', parentPath, elementName, vector, numElements, format)



    def updateFloatVector(self, path, vector, numElements, format):
        # input arg conversion
        _c_path = ctypes.c_char_p(encode_for_c(path))
        array_t_vector = ctypes.c_double * len(vector)
        _c_vector = array_t_vector(*vector)
        _c_numElements = ctypes.c_int(numElements)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiUpdateFloatVector(self._handle, _c_path, _c_vector, _c_numElements, _c_format)
        catch_error(errorCode, 'tixiUpdateFloatVector', path, vector, numElements, format)



    def createElement(self, parentPath, elementName):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))

        # call to native function
        errorCode = self.lib.tixiCreateElement(self._handle, _c_parentPath, _c_elementName)
        catch_error(errorCode, 'tixiCreateElement', parentPath, elementName)



    def createElementAtIndex(self, parentPath, elementName, index):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_index = ctypes.c_int(index)

        # call to native function
        errorCode = self.lib.tixiCreateElementAtIndex(self._handle, _c_parentPath, _c_elementName, _c_index)
        catch_error(errorCode, 'tixiCreateElementAtIndex', parentPath, elementName, index)



    def removeElement(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # call to native function
        errorCode = self.lib.tixiRemoveElement(self._handle, _c_elementPath)
        catch_error(errorCode, 'tixiRemoveElement', elementPath)



    def getNodeType(self, nodePath):
        # input arg conversion
        _c_nodePath = ctypes.c_char_p(encode_for_c(nodePath))

        # output arg preparation
        _c_nodeType = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetNodeType(self._handle, _c_nodePath, ctypes.byref(_c_nodeType))
        catch_error(errorCode, 'tixiGetNodeType', nodePath)

        _py_nodeType = decode_for_py(_c_nodeType.value)

        return _py_nodeType



    def getNamedChildrenCount(self, elementPath, childName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_childName = ctypes.c_char_p(encode_for_c(childName))

        # output arg preparation
        _c_count = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetNamedChildrenCount(self._handle, _c_elementPath, _c_childName, ctypes.byref(_c_count))
        catch_error(errorCode, 'tixiGetNamedChildrenCount', elementPath, childName)

        _py_count = _c_count.value

        return _py_count



    def getChildNodeName(self, parentElementPath, index):
        # input arg conversion
        _c_parentElementPath = ctypes.c_char_p(encode_for_c(parentElementPath))
        _c_index = ctypes.c_int(index)

        # output arg preparation
        _c_name = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetChildNodeName(self._handle, _c_parentElementPath, _c_index, ctypes.byref(_c_name))
        catch_error(errorCode, 'tixiGetChildNodeName', parentElementPath, index)

        _py_name = decode_for_py(_c_name.value)

        return _py_name



    def getNumberOfChilds(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_nChilds = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetNumberOfChilds(self._handle, _c_elementPath, ctypes.byref(_c_nChilds))
        catch_error(errorCode, 'tixiGetNumberOfChilds', elementPath)

        _py_nChilds = _c_nChilds.value

        return _py_nChilds



    def getTextAttribute(self, elementPath, attributeName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))

        # output arg preparation
        _c_text = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetTextAttribute(self._handle, _c_elementPath, _c_attributeName, ctypes.byref(_c_text))
        catch_error(errorCode, 'tixiGetTextAttribute', elementPath, attributeName)

        _py_text = decode_for_py(_c_text.value)

        return _py_text



    def getIntegerAttribute(self, elementPath, attributeName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))

        # output arg preparation
        _c_number = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetIntegerAttribute(self._handle, _c_elementPath, _c_attributeName, ctypes.byref(_c_number))
        catch_error(errorCode, 'tixiGetIntegerAttribute', elementPath, attributeName)

        _py_number = _c_number.value

        return _py_number



    def getBooleanAttribute(self, elementPath, attributeName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))

        # output arg preparation
        _c_boolean = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetBooleanAttribute(self._handle, _c_elementPath, _c_attributeName, ctypes.byref(_c_boolean))
        catch_error(errorCode, 'tixiGetBooleanAttribute', elementPath, attributeName)

        _py_boolean = _c_boolean.value

        return _py_boolean



    def getDoubleAttribute(self, elementPath, attributeName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))

        # output arg preparation
        _c_number = ctypes.c_double()

        # call to native function
        errorCode = self.lib.tixiGetDoubleAttribute(self._handle, _c_elementPath, _c_attributeName, ctypes.byref(_c_number))
        catch_error(errorCode, 'tixiGetDoubleAttribute', elementPath, attributeName)

        _py_number = _c_number.value

        return _py_number



    def addTextAttribute(self, elementPath, attributeName, attributeValue):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))
        _c_attributeValue = ctypes.c_char_p(encode_for_c(attributeValue))

        # call to native function
        errorCode = self.lib.tixiAddTextAttribute(self._handle, _c_elementPath, _c_attributeName, _c_attributeValue)
        catch_error(errorCode, 'tixiAddTextAttribute', elementPath, attributeName, attributeValue)



    def addDoubleAttribute(self, elementPath, attributeName, number, format):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))
        _c_number = ctypes.c_double(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddDoubleAttribute(self._handle, _c_elementPath, _c_attributeName, _c_number, _c_format)
        catch_error(errorCode, 'tixiAddDoubleAttribute', elementPath, attributeName, number, format)



    def addIntegerAttribute(self, elementPath, attributeName, number, format):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))
        _c_number = ctypes.c_int(number)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddIntegerAttribute(self._handle, _c_elementPath, _c_attributeName, _c_number, _c_format)
        catch_error(errorCode, 'tixiAddIntegerAttribute', elementPath, attributeName, number, format)



    def removeAttribute(self, elementPath, attributeName):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attributeName = ctypes.c_char_p(encode_for_c(attributeName))

        # call to native function
        errorCode = self.lib.tixiRemoveAttribute(self._handle, _c_elementPath, _c_attributeName)
        catch_error(errorCode, 'tixiRemoveAttribute', elementPath, attributeName)



    def getNumberOfAttributes(self, elementPath):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))

        # output arg preparation
        _c_nAttributes = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetNumberOfAttributes(self._handle, _c_elementPath, ctypes.byref(_c_nAttributes))
        catch_error(errorCode, 'tixiGetNumberOfAttributes', elementPath)

        _py_nAttributes = _c_nAttributes.value

        return _py_nAttributes



    def getAttributeName(self, elementPath, attrIndex):
        # input arg conversion
        _c_elementPath = ctypes.c_char_p(encode_for_c(elementPath))
        _c_attrIndex = ctypes.c_int(attrIndex)

        # output arg preparation
        _c_attrName = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetAttributeName(self._handle, _c_elementPath, _c_attrIndex, ctypes.byref(_c_attrName))
        catch_error(errorCode, 'tixiGetAttributeName', elementPath, attrIndex)

        _py_attrName = decode_for_py(_c_attrName.value)

        return _py_attrName



    def addExternalLink(self, parentPath, url, fileFormat):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_url = ctypes.c_char_p(encode_for_c(url))
        _c_fileFormat = ctypes.c_char_p(encode_for_c(fileFormat))

        # call to native function
        errorCode = self.lib.tixiAddExternalLink(self._handle, _c_parentPath, _c_url, _c_fileFormat)
        catch_error(errorCode, 'tixiAddExternalLink', parentPath, url, fileFormat)



    def addHeader(self, toolName, version, authorName):
        # input arg conversion
        _c_toolName = ctypes.c_char_p(encode_for_c(toolName))
        _c_version = ctypes.c_char_p(encode_for_c(version))
        _c_authorName = ctypes.c_char_p(encode_for_c(authorName))

        # call to native function
        errorCode = self.lib.tixiAddHeader(self._handle, _c_toolName, _c_version, _c_authorName)
        catch_error(errorCode, 'tixiAddHeader', toolName, version, authorName)



    def addCpacsHeader(self, name, creator, version, description, cpacsVersion):
        # input arg conversion
        _c_name = ctypes.c_char_p(encode_for_c(name))
        _c_creator = ctypes.c_char_p(encode_for_c(creator))
        _c_version = ctypes.c_char_p(encode_for_c(version))
        _c_description = ctypes.c_char_p(encode_for_c(description))
        _c_cpacsVersion = ctypes.c_char_p(encode_for_c(cpacsVersion))

        # call to native function
        errorCode = self.lib.tixiAddCpacsHeader(self._handle, _c_name, _c_creator, _c_version, _c_description, _c_cpacsVersion)
        catch_error(errorCode, 'tixiAddCpacsHeader', name, creator, version, description, cpacsVersion)



    def checkDocumentHandle(self):

        # call to native function
        errorCode = self.lib.tixiCheckDocumentHandle(self._handle)
        catch_error(errorCode, 'tixiCheckDocumentHandle')



    def usePrettyPrint(self, usePrettyPrint):
        # input arg conversion
        _c_usePrettyPrint = ctypes.c_int(usePrettyPrint)

        # call to native function
        errorCode = self.lib.tixiUsePrettyPrint(self._handle, _c_usePrettyPrint)
        catch_error(errorCode, 'tixiUsePrettyPrint', usePrettyPrint)



    def getPrintMsgFunc(self):

        # call to native function
        self.lib.tixiGetPrintMsgFunc.restype = ctypes.c_void
        _c_ret = self.lib.tixiGetPrintMsgFunc()


        return _py_ret



    def addDoubleListWithAttributes(self, parentPath, listName, childName, childAttributeName, values, format, attributes, nValues):
        # input arg conversion
        _c_parentPath = ctypes.c_char_p(encode_for_c(parentPath))
        _c_listName = ctypes.c_char_p(encode_for_c(listName))
        _c_childName = ctypes.c_char_p(encode_for_c(childName))
        _c_childAttributeName = ctypes.c_char_p(encode_for_c(childAttributeName))
        array_t_values = ctypes.c_double * len(values)
        _c_values = array_t_values(*values)
        _c_format = ctypes.c_char_p(encode_for_c(format))
        array_t_attributes = ctypes.c_char_p * len(attributes)
        _c_attributes = array_t_attributes()
        for i in range(len(attributes)):
            _c_attributes[i] = encode_for_c(attributes[i])
        _c_nValues = ctypes.c_int(nValues)

        # call to native function
        errorCode = self.lib.tixiAddDoubleListWithAttributes(self._handle, _c_parentPath, _c_listName, _c_childName, _c_childAttributeName, _c_values, _c_format, _c_attributes, _c_nValues)
        catch_error(errorCode, 'tixiAddDoubleListWithAttributes', parentPath, listName, childName, childAttributeName, values, format, attributes, nValues)



    def getVectorSize(self, vectorPath):
        # input arg conversion
        _c_vectorPath = ctypes.c_char_p(encode_for_c(vectorPath))

        # output arg preparation
        _c_nElements = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetVectorSize(self._handle, _c_vectorPath, ctypes.byref(_c_nElements))
        catch_error(errorCode, 'tixiGetVectorSize', vectorPath)

        _py_nElements = _c_nElements.value

        return _py_nElements



    def getFloatVector(self, vectorPath, eNumber):
        # input arg conversion
        _c_vectorPath = ctypes.c_char_p(encode_for_c(vectorPath))
        _c_eNumber = ctypes.c_int(eNumber)

        # output arg preparation
        _c_vectorArray = ctypes.POINTER(ctypes.c_double)()

        # call to native function
        errorCode = self.lib.tixiGetFloatVector(self._handle, _c_vectorPath, ctypes.byref(_c_vectorArray), _c_eNumber)
        catch_error(errorCode, 'tixiGetFloatVector', vectorPath, eNumber)

        vectorArray_array_size = eNumber 
        _py_vectorArray = tuple(_c_vectorArray[i] for i in range(vectorArray_array_size))

        return _py_vectorArray



    def getArrayDimensions(self, arrayPath):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))

        # output arg preparation
        _c_dimensions = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetArrayDimensions(self._handle, _c_arrayPath, ctypes.byref(_c_dimensions))
        catch_error(errorCode, 'tixiGetArrayDimensions', arrayPath)

        _py_dimensions = _c_dimensions.value

        return _py_dimensions



    def getArrayDimensionSizes(self, arrayPath, sizes_len):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))

        # output arg preparation
        _c_sizes = (ctypes.c_int * sizes_len)()
        _c_linearArraySize = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetArrayDimensionSizes(self._handle, _c_arrayPath, ctypes.byref(_c_sizes), ctypes.byref(_c_linearArraySize))
        catch_error(errorCode, 'tixiGetArrayDimensionSizes', arrayPath)

        _py_linearArraySize = _c_linearArraySize.value
        sizes_array_size = sizes_len
        _py_sizes = tuple(_c_sizes[i] for i in range(sizes_array_size))

        return (_py_sizes, _py_linearArraySize)



    def getArrayDimensionNames(self, arrayPath, dimensionNames_len):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))

        # output arg preparation
        _c_dimensionNames = (ctypes.c_char_p * dimensionNames_len)()

        # call to native function
        errorCode = self.lib.tixiGetArrayDimensionNames(self._handle, _c_arrayPath, ctypes.byref(_c_dimensionNames))
        catch_error(errorCode, 'tixiGetArrayDimensionNames', arrayPath)

        dimensionNames_array_size = dimensionNames_len
        _py_dimensionNames = tuple(decode_for_py(_c_dimensionNames[i]) for i in range(dimensionNames_array_size))

        return _py_dimensionNames



    def getArrayDimensionValues(self, arrayPath, dimension, dimensionValues_len):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))
        _c_dimension = ctypes.c_int(dimension)

        # output arg preparation
        _c_dimensionValues = (ctypes.c_double * dimensionValues_len)()

        # call to native function
        errorCode = self.lib.tixiGetArrayDimensionValues(self._handle, _c_arrayPath, _c_dimension, ctypes.byref(_c_dimensionValues))
        catch_error(errorCode, 'tixiGetArrayDimensionValues', arrayPath, dimension)

        dimensionValues_array_size = dimensionValues_len
        _py_dimensionValues = tuple(_c_dimensionValues[i] for i in range(dimensionValues_array_size))

        return _py_dimensionValues



    def getArrayParameters(self, arrayPath):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))

        # output arg preparation
        _c_parameters = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetArrayParameters(self._handle, _c_arrayPath, ctypes.byref(_c_parameters))
        catch_error(errorCode, 'tixiGetArrayParameters', arrayPath)

        _py_parameters = _c_parameters.value

        return _py_parameters



    def getArrayParameterNames(self, arrayPath, parameterNames_len):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))

        # output arg preparation
        _c_parameterNames = (ctypes.c_char_p * parameterNames_len)()

        # call to native function
        errorCode = self.lib.tixiGetArrayParameterNames(self._handle, _c_arrayPath, ctypes.byref(_c_parameterNames))
        catch_error(errorCode, 'tixiGetArrayParameterNames', arrayPath)

        parameterNames_array_size = parameterNames_len
        _py_parameterNames = tuple(decode_for_py(_c_parameterNames[i]) for i in range(parameterNames_array_size))

        return _py_parameterNames



    def getArray(self, arrayPath, elementName, arraySize):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))
        _c_arraySize = ctypes.c_int(arraySize)

        # output arg preparation
        _c_values = ctypes.POINTER(ctypes.c_double)()

        # call to native function
        errorCode = self.lib.tixiGetArray(self._handle, _c_arrayPath, _c_elementName, _c_arraySize, ctypes.byref(_c_values))
        catch_error(errorCode, 'tixiGetArray', arrayPath, elementName, arraySize)

        values_array_size = arraySize 
        _py_values = tuple(_c_values[i] for i in range(values_array_size))

        return _py_values



    def getArrayValue(self, array, dimSize, dimPos, dims):
        # input arg conversion
        array_t_array = ctypes.c_double * len(array)
        _c_array = array_t_array(*array)
        array_t_dimSize = ctypes.c_int * len(dimSize)
        _c_dimSize = array_t_dimSize(*dimSize)
        array_t_dimPos = ctypes.c_int * len(dimPos)
        _c_dimPos = array_t_dimPos(*dimPos)
        _c_dims = ctypes.c_int(dims)

        # call to native function
        self.lib.tixiGetArrayValue.restype = ctypes.c_double
        _c_ret = self.lib.tixiGetArrayValue(_c_array, _c_dimSize, _c_dimPos, _c_dims)


        return _py_ret



    def getArrayElementCount(self, arrayPath, elementName):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))
        _c_elementName = ctypes.c_char_p(encode_for_c(elementName))

        # output arg preparation
        _c_elements = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiGetArrayElementCount(self._handle, _c_arrayPath, _c_elementName, ctypes.byref(_c_elements))
        catch_error(errorCode, 'tixiGetArrayElementCount', arrayPath, elementName)

        _py_elements = _c_elements.value

        return _py_elements



    def getArrayElementNames(self, arrayPath, elementType):
        # input arg conversion
        _c_arrayPath = ctypes.c_char_p(encode_for_c(arrayPath))
        _c_elementType = ctypes.c_char_p(encode_for_c(elementType))

        # output arg preparation
        _c_elementNames = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiGetArrayElementNames(self._handle, _c_arrayPath, _c_elementType, ctypes.byref(_c_elementNames))
        catch_error(errorCode, 'tixiGetArrayElementNames', arrayPath, elementType)

        _py_elementNames = decode_for_py(_c_elementNames.value)

        return _py_elementNames



    def addPoint(self, pointParentPath, x, y, z, format):
        # input arg conversion
        _c_pointParentPath = ctypes.c_char_p(encode_for_c(pointParentPath))
        _c_x = ctypes.c_double(x)
        _c_y = ctypes.c_double(y)
        _c_z = ctypes.c_double(z)
        _c_format = ctypes.c_char_p(encode_for_c(format))

        # call to native function
        errorCode = self.lib.tixiAddPoint(self._handle, _c_pointParentPath, _c_x, _c_y, _c_z, _c_format)
        catch_error(errorCode, 'tixiAddPoint', pointParentPath, x, y, z, format)



    def getPoint(self, pointParentPath):
        # input arg conversion
        _c_pointParentPath = ctypes.c_char_p(encode_for_c(pointParentPath))

        # output arg preparation
        _c_x = ctypes.c_double()
        _c_y = ctypes.c_double()
        _c_z = ctypes.c_double()

        # call to native function
        errorCode = self.lib.tixiGetPoint(self._handle, _c_pointParentPath, ctypes.byref(_c_x), ctypes.byref(_c_y), ctypes.byref(_c_z))
        catch_error(errorCode, 'tixiGetPoint', pointParentPath)

        _py_x = _c_x.value
        _py_y = _c_y.value
        _py_z = _c_z.value

        return (_py_x, _py_y, _py_z)



    def xSLTransformationToFile(self, xslFilename, resultFilename):
        # input arg conversion
        _c_xslFilename = ctypes.c_char_p(encode_for_c(xslFilename))
        _c_resultFilename = ctypes.c_char_p(encode_for_c(resultFilename))

        # call to native function
        errorCode = self.lib.tixiXSLTransformationToFile(self._handle, _c_xslFilename, _c_resultFilename)
        catch_error(errorCode, 'tixiXSLTransformationToFile', xslFilename, resultFilename)



    def xPathEvaluateNodeNumber(self, xPathExpression):
        # input arg conversion
        _c_xPathExpression = ctypes.c_char_p(encode_for_c(xPathExpression))

        # output arg preparation
        _c_number = ctypes.c_int()

        # call to native function
        errorCode = self.lib.tixiXPathEvaluateNodeNumber(self._handle, _c_xPathExpression, ctypes.byref(_c_number))
        catch_error(errorCode, 'tixiXPathEvaluateNodeNumber', xPathExpression)

        _py_number = _c_number.value

        return _py_number



    def xPathExpressionGetXPath(self, xPathExpression, index):
        # input arg conversion
        _c_xPathExpression = ctypes.c_char_p(encode_for_c(xPathExpression))
        _c_index = ctypes.c_int(index)

        # output arg preparation
        _c_xPath = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiXPathExpressionGetXPath(self._handle, _c_xPathExpression, _c_index, ctypes.byref(_c_xPath))
        catch_error(errorCode, 'tixiXPathExpressionGetXPath', xPathExpression, index)

        _py_xPath = decode_for_py(_c_xPath.value)

        return _py_xPath



    def xPathExpressionGetTextByIndex(self, xPathExpression, elementNumber):
        # input arg conversion
        _c_xPathExpression = ctypes.c_char_p(encode_for_c(xPathExpression))
        _c_elementNumber = ctypes.c_int(elementNumber)

        # output arg preparation
        _c_text = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiXPathExpressionGetTextByIndex(self._handle, _c_xPathExpression, _c_elementNumber, ctypes.byref(_c_text))
        catch_error(errorCode, 'tixiXPathExpressionGetTextByIndex', xPathExpression, elementNumber)

        _py_text = decode_for_py(_c_text.value)

        return _py_text



    def uIDCheckDuplicates(self):

        # call to native function
        errorCode = self.lib.tixiUIDCheckDuplicates(self._handle)
        catch_error(errorCode, 'tixiUIDCheckDuplicates')



    def uIDCheckLinks(self):

        # call to native function
        errorCode = self.lib.tixiUIDCheckLinks(self._handle)
        catch_error(errorCode, 'tixiUIDCheckLinks')



    def uIDGetXPath(self, uID):
        # input arg conversion
        _c_uID = ctypes.c_char_p(encode_for_c(uID))

        # output arg preparation
        _c_xPath = ctypes.c_char_p()

        # call to native function
        errorCode = self.lib.tixiUIDGetXPath(self._handle, _c_uID, ctypes.byref(_c_xPath))
        catch_error(errorCode, 'tixiUIDGetXPath', uID)

        _py_xPath = decode_for_py(_c_xPath.value)

        return _py_xPath



    def uIDSetToXPath(self, xPath, uID):
        # input arg conversion
        _c_xPath = ctypes.c_char_p(encode_for_c(xPath))
        _c_uID = ctypes.c_char_p(encode_for_c(uID))

        # call to native function
        errorCode = self.lib.tixiUIDSetToXPath(self._handle, _c_xPath, _c_uID)
        catch_error(errorCode, 'tixiUIDSetToXPath', xPath, uID)



