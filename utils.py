import json


def custom_serialize(obj, indent=2, current_level=0):
    if isinstance(obj, list):
        current_level += 1
        if any(isinstance(item, (list, dict)) for item in obj):
            # 包含嵌套结构，不是最后一级
            indent_str = ' ' * (indent * current_level)
            items = [f"{indent_str}{custom_serialize(item, indent, current_level)}" for item in obj]
            outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
            return '[\n' + ',\n'.join(items) + '\n' + outer_indent + ']'
        else:
            # 最后一级，都是基本类型
            return '[' + ', '.join(json.dumps(item, ensure_ascii=False) for item in obj) + ']'
    
    elif isinstance(obj, dict):
        current_level += 1
        indent_str = ' ' * (indent * current_level)
        items = []
        for key, value in obj.items():
            serialized_value = custom_serialize(value, indent, current_level)
            items.append(f'{indent_str}{json.dumps(key, ensure_ascii=False)}: {serialized_value}')
        
        outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
        return '{\n' + ',\n'.join(items) + '\n' + outer_indent + '}'
    
    else:
        return json.dumps(obj, ensure_ascii=False)
    