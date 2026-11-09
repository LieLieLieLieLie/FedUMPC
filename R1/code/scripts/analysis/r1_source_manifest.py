"""Record the exact evaluated local R1 snapshot without implying a public commit."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
BASE=R1_ROOT.parents[1]
def main():
    roots=[BASE/'code/R1',BASE/'MuJoCo/R1',BASE/'paper/IoTJ_FedUMPC/R1']
    files={}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(BASE).as_posix()
            if any(part.startswith('_superseded') or part=='__pycache__' for part in path.parts):continue
            if path.name in ['source_manifest.json','source_manifest.sha256']:continue
            if path.suffix.lower() not in ['.py','.tex','.bib','.md','.json','.csv','.pt','.npz','.xml','.pdf','.docx','.txt','.ps1']:continue
            files[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    packages={}
    for package in ['numpy','pandas','torch','matplotlib','scipy','scikit-learn','mujoco']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]='not installed in analytical runtime'
    output=BASE/'paper/IoTJ_FedUMPC/R1'
    physical_python=Path(r'E:\MuJoCo\runtime\Scripts\python.exe')
    physical={'executable':str(physical_python),'inventory_status':'runtime unavailable at capture'}
    if physical_python.exists():
        query=("import importlib.metadata as m,json,platform,sys; "
               "print(json.dumps({'executable':sys.executable,"
               "'python':platform.python_version(),'packages':"
               "{p:m.version(p) for p in ['mujoco','numpy','matplotlib','scipy','pillow']}}))")
        inventory=subprocess.run([str(physical_python),'-c',query],capture_output=True,text=True,check=True)
        physical=json.loads(inventory.stdout)
    content={'version':'FedUMPC-IoTJ-R1-20261107','original_submission':'IoT-72582-2026',
             'scope':'local supplied R1 snapshot; not an asserted public Git commit',
             'environment_inventory_note':'Installed versions checked after execution; analytical and MuJoCo runtimes are separate.',
             'analytical_runtime':{'executable':sys.executable,'python':platform.python_version(),'packages':packages},
             'mujoco_runtime':physical,'sha256':files}
    (output/'source_manifest.json').write_text(json.dumps(content,indent=2),encoding='utf-8')
    (output/'source_manifest.sha256').write_text('\n'.join(v+'  '+k for k,v in files.items())+'\n',encoding='utf-8')
    print('Hashed',len(files),'R1 source and evidence files')
if __name__=='__main__':main()
