"""Start the NEW launcher from a temporary cwd; check HTTP locally; stop only that child."""
import os,socket,subprocess,tempfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 with tempfile.TemporaryDirectory(prefix='hoi-ui-smoke-') as directory:
  root=Path(directory)
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  env=os.environ.copy();env.update(HOI_DATA_ROOT=str(root/'data'),WEBUI_RUNS_DIR=str(root/'runs'),WEBUI_HOST='127.0.0.1',WEBUI_PORT=str(port),WEBUI_SHARE='0',WEBUI_NO_AUTH='1')
  with (root/'server.log').open('w+') as log:
   process=subprocess.Popen(['bash',str(ROOT/'scripts/run_webui.sh')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
     if process.poll() is not None:log.seek(0);raise RuntimeError('WebUI exited: '+log.read())
     try:
      with urllib.request.urlopen(f'http://127.0.0.1:{port}/config',timeout=2) as response:
       import json
       config=json.load(response);assert len(config['components'])>250
      print('PASS: packaged WebUI HTTP /config,',len(config['components']),'components, launcher from arbitrary cwd, local-only, no datasets/models needed.');return
     except (OSError,urllib.error.URLError):time.sleep(.5)
    log.seek(0);raise RuntimeError('Startup deadline exceeded: '+log.read())
   finally:
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
