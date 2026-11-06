"""Git clean filter: read a notebook on stdin, write it without outputs on stdout.

Outputs and execution counts are removed; sources, cell tags and metadata are
kept. Used through .gitattributes so committed notebooks stay small while the
working copy keeps its outputs.
"""
import json
import sys


def main():
    notebook = json.load(sys.stdin)
    # Run-specific notebook metadata (local interpreter paths, run notes).
    for key in ('execution_environment', 'vscode', 'widgets'):
        notebook.get('metadata', {}).pop(key, None)
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        cell.get('metadata', {}).pop('execution', None)
    json.dump(notebook, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
