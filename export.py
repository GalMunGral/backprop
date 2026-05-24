import nbformat
from nbconvert import HTMLExporter
from pathlib import Path

nb = nbformat.read('backprop.ipynb', as_version=4)
Path('_site').mkdir(exist_ok=True)
body, _ = HTMLExporter().from_notebook_node(nb)
Path('_site/index.html').write_text(body)
print('wrote _site/index.html')