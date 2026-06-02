import ast
import os

for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check for pd.DataFrame or pd.Series
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "pd":
                            if node.func.attr in ("DataFrame", "Series"):
                                print(f"{filepath}:{node.lineno} - pd.{node.func.attr}")
                    elif isinstance(node.func, ast.Name):
                        if node.func.id in ("DataFrame", "Series"):
                            print(f"{filepath}:{node.lineno} - {node.func.id}")
