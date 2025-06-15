import re
from collections import defaultdict

def parse_formula_nested(formula: str) -> dict:
    stack = [defaultdict(int)]
    i = 0
    while i < len(formula):
        char = formula[i]
        if char in '([{':
            stack.append(defaultdict(int))
            i += 1
        elif char in ')]}':
            group_counts = stack.pop()
            i += 1
            # 解析括号后的数字
            num = ''
            while i < len(formula) and formula[i].isdigit():
                num += formula[i]
                i += 1
            multiplier = int(num) if num else 1
            for el, cnt in group_counts.items():
                stack[-1][el] += cnt * multiplier
        elif re.match(r'[A-Z]', char):
            match = re.match(r'([A-Z][a-z]?)(\d*)', formula[i:])
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1
            stack[-1][element] += count
            i += len(match.group(0))
        else:
            i += 1  # 跳过其他字符
    return dict(stack[-1])

examples = [
    "Mg[Fe(CN)6]2",
    "K4[ON(SO3)2]2",
    "Na3[Co(NO2)6]",
    "C2H5OH"
]

for f in examples:
    print(f"{f} → {parse_formula_nested(f)}")
