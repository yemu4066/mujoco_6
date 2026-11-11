#!/usr/bin/env python3
"""静态检查：函数里有没有"引用了但没定义"的名字。

专门防这一类低级错误：
    函数体里用了 car_length，但构造函数签名里忘了加
    → 语法检查（ast.parse）不报错
    → import 也不报错
    → 一实例化就 NameError
    → 整个节点起不来，/costmap 消失，Foxglove 空白

用法：
    python3 tests/check_names.py
"""
import ast
import builtins
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = ['nav/navigation.py', 'nav/path_smooth.py', 'nav/map.py',
         'nav/astar.py', 'nav/grid.py',
         'nav/mpc_core/mpc_acados.py', 'nav/mpc_core/linear.py']


def collect_module_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def check_function(fn, outer, path):
    """检查单个函数；outer 是它能看到的闭包/外层名字。"""
    local = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                local.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, ast.arg):
            local.add(node.arg)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    local.add(t.id)

    known = local | outer | set(dir(builtins))
    bad = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known and node.id != 'self':
                bad.append((node.lineno, node.id))
    return [(fn.name, ln, nm) for ln, nm in bad]


def scan(rel):
    path = ROOT / rel
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module_names = collect_module_names(tree)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 嵌套函数：把外层函数的名字也当作可见
            problems.extend(check_function(node, module_names, rel))
    return problems


def main():
    failed = False
    for rel in FILES:
        problems = scan(rel)
        if problems:
            failed = True
            print(f'✗ {rel}')
            for fn, ln, nm in problems:
                print(f'    第 {ln:4d} 行  {fn}()  引用了未定义的 `{nm}`')
        else:
            print(f'✓ {rel}')
    if failed:
        print('\n有未定义变量，实例化时会 NameError。')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
