import base64
import json

from google.protobuf import json_format, descriptor


# 通用json序列化
def json_dumps(obj) -> str:
    return json.dumps(obj, default=str, separators=(',', ':'))


# proto json序列化
def proto_json_dumps(obj) -> str:
    return json.dumps(obj, default=proto_dict_format, separators=(',', ':'))


# proto 转 dict
def proto_dict_format(message) -> str:
    _dict = json_format.MessageToDict(message, including_default_value_fields=True, preserving_proto_field_name=True)
    for field, value in message.ListFields():
        if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING and field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            _dict[field.name] = base64.b64decode(_dict[field.name]).decode()
        if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            _dict[field.name] = [proto_dict_format(k) for k in value]
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE and field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
            _dict[field.name] = proto_dict_format(value)
    return _dict


# proto json序列化2
def proto_json_dumps2(obj) -> str:
    return json.dumps(obj, default=proto_json_format, separators=(',', ':'))


# proto 转 json
def proto_json_format(message) -> str:
    return json_format.MessageToJson(message, including_default_value_fields=True, preserving_proto_field_name=True)
