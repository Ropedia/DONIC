'use strict';
const $ = (id) => document.getElementById(id);
const states = Object.fromEntries(['wapr','mose3'].map(app => [app, {files:{}, values:{}, job:null}]));
let app = 'wapr', config, image, imageURL, videoURL, drawing, uploadAbort, queryImage;
const uploadCache = new WeakMap();
const definitions = {
  wapr: {name:'WAPR', tag:'POSE ESTIMATION', eyebrow:'RGB-D → 6D POSE', title:'让物体的位姿清晰可见',
    subtitle:'输入 RGB-D 与物体网格，估计相机坐标系中的旋转和平移，并在原图中查看位姿。',
    repo:'https://github.com/WangYuLin-SEU/WAPR', action:'开始位姿估计', preview:'RGB 图像',
    empty:'从一张 RGB 图像开始', emptyText:'上传后预览图像，并选择目标物体区域', output:'4 × 4 位姿矩阵 · 叠加图 · PLY',
    guide:'深度、网格和标定需要使用一致的尺度',
    guideText:'深度与网格会按所填系数转换为米。相机内参应对应原始 RGB 分辨率；Mask 与深度须与 RGB 同尺寸。直径填写真实物体直径（米），结果分数是排序信号。'},
  mose3: {name:'MoSE3', tag:'DENSE MOTION TRACKING', eyebrow:'RGB VIDEO → WORLD-SPACE SE(3)', title:'从每个像素，看见三维运动',
    subtitle:'上传一段 RGB 视频，预测稠密 3D 轨迹与逐像素刚体运动，探索物体如何移动。',
    repo:'https://github.com/mose3-tracker/MoSE3', action:'开始运动跟踪', preview:'输入视频',
    empty:'用一段视频记录运动', emptyText:'上传 RGB 视频，预览后设置采样与跟踪参数', output:'3D 轨迹 · SE(3) 视频 · NPZ',
    guide:'按 10 FPS 采样与播放',
    guideText:'每 0.1 秒采样一次，结果视频以 10 FPS 播放。选择查询帧并点击需要跟踪的位置；长视频会占用更多显存。世界坐标采用第一帧相机系，单目预测不保证米制尺度。'}
};
function uploadField(key,title,types,accept,wide=false) {
  return `<label class="upload-card ${wide?'wide':''}" data-upload="${key}"><input type="file" data-file="${key}" accept="${accept}" aria-label="${title}"><span class="upload-symbol">+</span><strong>${title}</strong><small>${types}</small></label>`;
}
function numberField(name,label,value,min,max,step='any',placeholder='') {
  return `<div class="field"><label for="${name}">${label}</label><input id="${name}" name="${name}" type="number" value="${value}" min="${min}" max="${max}" step="${step}" placeholder="${placeholder}" required></div>`;
}
function gpuField(){return '<div class="field"><label for="device">推理设备</label><input id="device" name="device" value="cuda:0" pattern="cuda(:[0-9]+)?" required></div>';}
function waprFields(){return `
  <div class="fields"><div class="field-label">观测与物体模型<span>拖入文件或点击上传</span></div><div class="upload-grid">
    ${uploadField('rgb','RGB 图像','PNG / JPG','.png,.jpg,.jpeg')}
    ${uploadField('depth','深度图','PNG / NPY / TIFF','.png,.npy,.tif,.tiff')}
    ${uploadField('mesh','物体网格','PLY / OBJ / STL / GLB','.ply,.obj,.stl,.glb')}
    ${uploadField('mask','前景 Mask','可选 · 非零值为前景','.png,.jpg,.jpeg')}
  </div></div><div class="form-divider"></div>
  <div class="fields"><h3 class="section-title">目标物体区域</h3><div class="segmented"><label><input type="radio" name="region" value="box" checked>拖动框选</label><label><input type="radio" name="region" value="mask">使用 Mask</label></div>
    <div id="bbox-fields"><div class="field-row">${numberField('x','X / 左上角',0,0,100000,1)}${numberField('y','Y / 左上角',0,0,100000,1)}</div><div class="field-row">${numberField('w','宽度 / px','',1,100000,1)}${numberField('h','高度 / px','',1,100000,1)}</div><p class="hint">在右侧原图上拖动框选，也可直接输入像素坐标。</p></div>
    <p id="mask-help" class="hint" hidden>上传黑底白前景 Mask；将使用所有非零像素。</p>
  </div><div class="form-divider"></div>
  <div class="fields"><h3 class="section-title">相机内参 <span class="hint">/ 原图像素单位</span></h3><div class="field-row">${numberField('fx','fx','',.000001,1e7,'any','焦距 x')}${numberField('fy','fy','',.000001,1e7,'any','焦距 y')}</div><div class="field-row">${numberField('cx','cx','',-1e7,1e7,'any','主点 x')}${numberField('cy','cy','',-1e7,1e7,'any','主点 y')}</div></div>
  <details class="advanced" open><summary>尺度与推理设置</summary><div class="field-row">${numberField('depth_scale','深度 → 米',.001,1e-10,1e6)}${numberField('mesh_scale','网格 → 米',.001,1e-10,1e6)}</div><p class="hint">毫米数据填 0.001；已经是米的数据填 1。</p><div class="field-row">${numberField('diameter_m','物体直径 / m','',1e-10,1e6,'any','例如 0.12')}${gpuField()}</div><div class="field-row"><div class="field"><label for="backend">推理后端</label><select id="backend" name="backend"><option value="torch">PyTorch</option><option value="trt">TensorRT</option></select></div></div><p class="hint">首次运行会下载缺失的官方权重。TensorRT 需已准备匹配的运行环境。</p></details>`;}
function mose3Fields(){return `
  <div class="fields"><div class="field-label">RGB 视频<span>拖入文件或点击上传</span></div><div class="upload-grid">${uploadField('video','选择一段视频','MP4 / MOV / AVI / WebM / MKV','.mp4,.mov,.avi,.webm,.mkv',true)}</div><p class="hint">仅需要 RGB 视频，无需深度、物体网格或相机标定。</p></div><div class="form-divider"></div>
  <div class="fields"><h3 class="section-title">视频帧</h3><div class="field-row"><div class="field"><label>采样帧率</label><input value="10 FPS" readonly></div><div class="field"><label>播放帧率</label><input value="10 FPS" readonly></div></div><input type="checkbox" id="sample_all" name="sample_all" hidden><input type="hidden" name="sampling_fps" value="10"><p id="all-frame-count" class="hint">每 0.1 秒采样一次；生成预览后显示采样帧数。</p><div class="field-row">${numberField('num_frames','采样帧数',24,2,300,1)}${numberField('resolution','最大边长 / px',280,112,1036,1)}</div><div class="field-row">${numberField('start_frame','起始帧',0,0,10000000,1)}${numberField('frame_interval','采样间隔',0,0,100000,1)}</div><div class="field-row">${numberField('query_idx','查询帧索引',0,0,299,1)}<div class="field"><label>采样间隔</label><input value="0.1 秒" readonly><input type="hidden" name="fps" value="10"></div></div><p class="hint">查询帧为 10 FPS 采样序列中的索引，从 0 开始。</p></div>
  <div class="fields"><h3 class="section-title">查询点</h3><div class="segmented"><label><input type="radio" name="point_mode" value="manual" checked>手动点击</label><label><input type="radio" name="point_mode" value="grid">自动网格</label></div><button type="button" id="prepare-points" class="secondary">生成帧预览并选点</button><p class="hint" id="selection-hint">生成预览后选择一帧，直接点击图像添加查询点，最多 128 个。切换查询帧会清空已选点。</p></div>
  <details class="advanced"><summary>运动拟合与可视化</summary><div class="field-row">${numberField('num_ref_pts','参考点数量',12000,64,50000,1)}${numberField('stride','显示网格间距 / px',48,8,256,1)}</div><div class="field-row">${numberField('se3_sigma_px','局部权重 σ / px',8,.1,1000)}${gpuField()}</div><label class="check"><input type="checkbox" name="se3_dist_weight">启用局部距离权重（适合布料等形变表面）</label><label class="check"><input type="checkbox" name="central_crop">先中心裁剪为正方形</label><p class="hint">默认使用官方 Hugging Face 权重，首次运行会下载并缓存。</p></details>`;}
function values(){return Object.fromEntries([...$('form').querySelectorAll('input[name],select[name]')].filter(el=>el.type!=='radio'||el.checked).map(el=>[el.name,el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value]));}
function message(text,success=false){$('notice').textContent=text;$('notice').className=success?'notice success':'notice';$('notice').hidden=!text;}
function switchView(view){for(const name of ['input','result','log'])$(name+'-view').hidden=name!==view;document.querySelectorAll('[data-view]').forEach(el=>{const selected=el.dataset.view===view;el.classList.toggle('active',selected);el.setAttribute('aria-selected',String(selected));});}
function regionChanged(){const box=$('form').elements.region?.value==='box';$('bbox-fields').hidden=!box;$('mask-help').hidden=box;for(const key of ['x','y','w','h'])$(key).disabled=!box;$('rgb-canvas').classList.toggle('box-mode',box);draw();}
function render(next){
  if(uploadAbort)return;
  if(config?.app!=='all' && config?.app!==next){next=config.app;}
  if($('input-fields').children.length)states[app].values=values();
  app=next;const def=definitions[app];document.title=`${def.name} · Pose Lab`;
  for(const [id,value] of Object.entries({breadcrumb:def.name,eyebrow:def.eyebrow,subtitle:def.subtitle,'preview-label':def.preview,'empty-title':def.empty,'empty-text':def.emptyText,'output-type':def.output,'guide-title':def.guide,'guide-text':def.guideText}))$(id).textContent=value;
  $('title').replaceChildren(document.createTextNode(def.title),Object.assign(document.createElement('span'),{textContent:'。'}));
  $('model-tag').replaceChildren(document.createTextNode(def.name),Object.assign(document.createElement('span'),{textContent:def.tag}));
  $('repo-link').href=def.repo;
  document.querySelectorAll('[data-app]').forEach(el=>{el.classList.toggle('active',el.dataset.app===app);el.hidden=config.app!=='all'&&config.app!==el.dataset.app;});
  $('input-fields').innerHTML=app==='wapr'?waprFields():mose3Fields();
  for(const [name,value] of Object.entries(states[app].values))for(const el of $('form').querySelectorAll(`[name="${name}"]`)){if(el.type==='checkbox')el.checked=value;else if(el.type==='radio')el.checked=el.value===value;else el.value=value;}
  document.querySelectorAll('[data-file]').forEach(el=>el.addEventListener('change',()=>chooseFile(el.dataset.file,el.files[0])));
  document.querySelectorAll('[data-upload]').forEach(el=>{el.addEventListener('dragover',e=>{e.preventDefault();el.classList.add('drag');});el.addEventListener('dragleave',()=>el.classList.remove('drag'));el.addEventListener('drop',e=>{e.preventDefault();el.classList.remove('drag');chooseFile(el.dataset.upload,e.dataTransfer.files[0]);});});
  if(app==='wapr'){document.querySelectorAll('[name="region"]').forEach(el=>el.addEventListener('change',regionChanged));for(const key of ['x','y','w','h'])$(key).addEventListener('input',draw);regionChanged();}
  if(app==='mose3'){
    document.querySelectorAll('[name="point_mode"]').forEach(el=>el.addEventListener('change',showPointPicker));
    $('prepare-points').addEventListener('click',preparePoints);
    for(const key of ['num_frames','start_frame','frame_interval','resolution','central_crop'])$(key)?.addEventListener('change',invalidatePoints);
    // central_crop is a named checkbox rather than a numbered input.
    $('form').elements.central_crop.addEventListener('change',invalidatePoints);
    $('query_idx').addEventListener('change',()=>setQueryFrame(Number($('query_idx').value)));
    $('sample_all').addEventListener('change',()=>{invalidatePoints();configureSamplingMode();});
    configureSamplingMode();
  }
  $('environment').textContent=`服务已连接，模型可用性将在运行时检查。\n\nPython: ${config.python[app]}\n源码: ${config.repos[app]}\n\n通过启动参数指定对应模型的 Python 环境。`;
  $('run-note').textContent=app==='mose3'?'首次运行可能下载权重；高分辨率会增加显存使用。':'首次加载模型和渲染组件可能需要几分钟。';
  message('');image=null;$('rgb-canvas').hidden=true;$('input-video').hidden=true;$('input-media').querySelector('.empty').hidden=false;$('dimensions').textContent='尚未载入';
  for(const [key,file] of Object.entries(states[app].files))updateUpload(key,file);
  refreshPreview();showPointPicker();switchView('input');renderJob(states[app].job);
}
function updateUpload(key,file){const el=document.querySelector(`[data-upload="${key}"]`);if(!el)return;el.classList.add('loaded');el.querySelector('strong').textContent=file.name;el.querySelector('small').textContent=`${(file.size/1024/1024).toFixed(2)} MB · 点击更换`;el.querySelector('.upload-symbol').textContent='✓';}
function chooseFile(key,file){if(!file)return;if(file.size>config.max_upload){message('单个文件不能超过 2 GiB。');return;}states[app].files[key]=file;updateUpload(key,file);message('');if(key==='video')invalidatePoints();if(key==='rgb'||key==='video')refreshPreview();}
function refreshPreview(){
  if(imageURL){URL.revokeObjectURL(imageURL);imageURL=null;}if(videoURL){URL.revokeObjectURL(videoURL);videoURL=null;}
  const selectedApp=app, file=states[app].files[app==='wapr'?'rgb':'video'];if(!file)return;
  if(app==='wapr'){
    image=new Image();imageURL=URL.createObjectURL(file);const pending=image;
    image.onload=()=>{if(app!==selectedApp||image!==pending)return;$('input-media').querySelector('.empty').hidden=true;const canvas=$('rgb-canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.hidden=false;$('dimensions').textContent=`${canvas.width} × ${canvas.height} px`;$('preview-help').textContent='选择「拖动框选」，在图像上拖出目标框。坐标始终对应原始图像。';draw();};
    image.onerror=()=>message('无法在浏览器读取这张 RGB 图像。');image.src=imageURL;
  }else{
    $('input-media').querySelector('.empty').hidden=true;const video=$('input-video');videoURL=URL.createObjectURL(file);video.src=videoURL;video.hidden=false;$('preview-help').textContent='这里预览完整输入视频；推理使用所设置的采样帧。浏览器可能不支持部分视频编码。';
    video.onloadedmetadata=()=>{if(app==='mose3')$('dimensions').textContent=`${video.videoWidth} × ${video.videoHeight} · ${video.duration.toFixed(1)} s`;};
    video.onerror=()=>message('浏览器无法预览此视频编码；可以转为 H.264 MP4，或继续交给模型读取。');
  }
}
function draw(){if(app!=='wapr'||!image?.complete||!image.naturalWidth)return;const canvas=$('rgb-canvas'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);if($('form').elements.region.value!=='box')return;const [x,y,w,h]=['x','y','w','h'].map(key=>Number($(key).value));if(w>0&&h>0){ctx.lineWidth=Math.max(2,canvas.width/350);ctx.fillStyle='#d6f38924';ctx.strokeStyle='#d6f389';ctx.fillRect(x,y,w,h);ctx.strokeRect(x,y,w,h);}}
function point(event){const r=$('rgb-canvas').getBoundingClientRect(),c=$('rgb-canvas');return [Math.max(0,Math.min(c.width,Math.round((event.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((event.clientY-r.top)*c.height/r.height)))];}
$('rgb-canvas').addEventListener('pointerdown',e=>{if(app!=='wapr'||$('form').elements.region.value!=='box')return;drawing=point(e);e.currentTarget.setPointerCapture(e.pointerId);});
$('rgb-canvas').addEventListener('pointermove',e=>{if(!drawing)return;const p=point(e);const vals=[Math.min(drawing[0],p[0]),Math.min(drawing[1],p[1]),Math.abs(p[0]-drawing[0]),Math.abs(p[1]-drawing[1])];['x','y','w','h'].forEach((key,i)=>$(key).value=vals[i]);draw();});
for(const event of ['pointerup','pointercancel'])$('rgb-canvas').addEventListener(event,()=>drawing=null);
async function api(path, options={}){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw Error(data.error||`请求失败 (${response.status})`);return data;}
function samplingKey(){const v=values(),f=states.mose3.files.video;return JSON.stringify([f?.name,f?.size,f?.lastModified,10,v.resolution,v.central_crop]);}
function configureSamplingMode(){ $('sample_all').checked=false;$('num_frames').closest('.field').hidden=true;$('start_frame').closest('.field-row').hidden=true;for(const key of ['num_frames','start_frame','frame_interval'])$(key).disabled=true;$('num_frames').max=10000000;$('query_idx').max=10000000;$('all-frame-count').hidden=false;}
function invalidatePoints(){states.mose3.picker=null;queryImage=null;showPointPicker();}
function showPointPicker(){
  const manual=app==='mose3'&&$('form').elements.point_mode?.value==='manual',picker=states.mose3.picker;
  $('point-picker').hidden=!manual||!picker;$('input-media').hidden=manual&&!!picker;
  if(app!=='mose3')return;
  $('prepare-points').hidden=!manual;$('selection-hint').hidden=!manual;
  if(!manual||!picker)$('preview-help').textContent='这里预览完整输入视频；生成帧预览后可在实际采样帧上点击选点。';
  if(manual&&picker){$('input-video').pause();$('preview-help').textContent='点击添加，右键撤销最后一点。坐标对应模型实际处理的帧；切换帧会清空选点。';showQueryFrame();}
}
function setQueryFrame(index){const p=states.mose3.picker;if(!p)return;index=Math.max(0,Math.min(p.preview.frames.length-1,Math.trunc(index)||0));if(index!==p.frame)p.points=[];p.frame=index;$('query_idx').value=index;showQueryFrame();}
function showQueryFrame(){
  const p=states.mose3.picker;if(app!=='mose3'||!p)return;
  const preview=p.preview;$('num_frames').value=preview.frames.length;$('all-frame-count').textContent=`10 FPS，共 ${preview.frames.length} 帧。`; $('query-frame-slider').max=preview.frames.length-1;$('query-frame-slider').value=p.frame;
  $('query-frame-label').textContent=`采样帧 ${p.frame} / ${preview.frames.length-1} · ${(p.frame/10).toFixed(1)} 秒 · 原视频第 ${preview.frame_indices[p.frame]} 帧 · ${preview.width} × ${preview.height} px`;
  $('previous-frame').disabled=p.frame===0;$('next-frame').disabled=p.frame===preview.frames.length-1;
  const src=new URL(preview.frames[p.frame],location.href).href;
  if(queryImage?.src===src&&queryImage.complete){drawQueryPoints();return;}
  const img=new Image();queryImage=img;const canvas=$('query-canvas');canvas.width=preview.width;canvas.height=preview.height;
  canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
  img.onload=()=>{if(queryImage===img&&states.mose3.picker===p)drawQueryPoints();};img.onerror=()=>message('无法读取选点帧，请重新生成预览。');img.src=src;drawQueryPoints();
}
function drawQueryPoints(){
  const p=states.mose3.picker;if(app!=='mose3'||!p)return;const canvas=$('query-canvas'),ctx=canvas.getContext('2d');
  if(queryImage?.complete&&queryImage.naturalWidth){ctx.drawImage(queryImage,0,0,canvas.width,canvas.height);}
  const scale=Math.max(1,canvas.width/500);ctx.font=`${12*scale}px sans-serif`;ctx.lineWidth=2*scale;
  p.points.forEach(([u,v],i)=>{const x=u*canvas.width,y=v*canvas.height;ctx.beginPath();ctx.arc(x,y,4*scale,0,2*Math.PI);ctx.fillStyle='#d6f389';ctx.fill();ctx.strokeStyle='#173b2d';ctx.stroke();ctx.fillStyle='#d6f389';ctx.fillText(String(i+1),x+7*scale,y-6*scale);});
  $('point-count').textContent=`已选 ${p.points.length} 个点`;$('undo-point').disabled=!p.points.length;$('clear-points').disabled=!p.points.length;
  $('point-list').replaceChildren(...p.points.map(([u,v],i)=>{const el=document.createElement('button');el.type='button';el.textContent=`${i+1}: (${Math.min(canvas.width-1,Math.floor(u*canvas.width))}, ${Math.min(canvas.height-1,Math.floor(v*canvas.height))}) ×`;el.setAttribute('aria-label',`删除第 ${i+1} 个查询点`);el.onclick=()=>{p.points.splice(i,1);drawQueryPoints();};return el;}));
}
async function preparePoints(){
  const file=states.mose3.files.video;if(!file){message('请先上传视频。');return;}
  const key=samplingKey(),body={...values(),app:'mose3'},abort=new AbortController();uploadAbort=abort;renderJob(states[app].job);$('prepare-points').disabled=true;message('正在按实际推理参数解码预览帧…',true);
  try{
    let token=uploadCache.get(file);if(!token){token=(await api('/api/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name),'Content-Type':'application/octet-stream'},body:file,signal:abort.signal})).token;uploadCache.set(file,token);}
    const preview=await api('/api/mose3/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,video:token}),signal:abort.signal});
    if(key!==samplingKey())throw Error('采样参数已改变，请重新生成预览。');
    states.mose3.picker={preview,points:[],frame:Math.min(body.query_idx,preview.frames.length-1),key};$('query_idx').value=states.mose3.picker.frame;queryImage=null;showPointPicker();switchView('input');message('选择一帧后，点击图像添加查询点。',true);
  }catch(error){message(error.name==='AbortError'?'已取消帧预览。':error.message);}
  finally{uploadAbort=null;renderJob(states[app].job);$('prepare-points').disabled=false;}
}
$('query-frame-slider').addEventListener('input',e=>setQueryFrame(Number(e.target.value)));
$('previous-frame').onclick=()=>setQueryFrame(states.mose3.picker.frame-1);
$('next-frame').onclick=()=>setQueryFrame(states.mose3.picker.frame+1);
$('undo-point').onclick=()=>{states.mose3.picker.points.pop();drawQueryPoints();};
$('clear-points').onclick=()=>{states.mose3.picker.points=[];drawQueryPoints();};
$('query-canvas').addEventListener('click',e=>{const p=states.mose3.picker;if(!p||!queryImage?.complete||!queryImage.naturalWidth)return;if(p.points.length>=128){message('最多可选 128 个查询点。');return;}const r=e.currentTarget.getBoundingClientRect();p.points.push([Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]);drawQueryPoints();});
$('query-canvas').addEventListener('contextmenu',e=>{e.preventDefault();if(states.mose3.picker){states.mose3.picker.points.pop();drawQueryPoints();}});
function renderJob(job){
  const running=job&&['queued','running'].includes(job.status);$('run').disabled=!!running||!!uploadAbort;$('reset').disabled=!!running||!!uploadAbort;$('example').disabled=!!running||!!uploadAbort;$('cancel').hidden=!running&&!uploadAbort;
  $('run').innerHTML='<span>▶</span> '+definitions[app].action;
  $('status').className='status '+(job?.status||'');$('status').textContent=({queued:'等待执行',running:'推理进行中',succeeded:'推理完成',failed:'推理失败',cancelled:'已取消'})[job?.status]||'等待输入';
  $('log').textContent=job?.log||'暂无任务。开始推理后，日志会显示在这里。';
  $('result-empty').hidden=job?.status==='succeeded';$('result-content').hidden=job?.status!=='succeeded';
  if(job?.status==='succeeded')renderResult(job);
}
async function renderResult(job){
  const resultApp=app;
  try{const latest=await api(`/files/${job.id}/result.json?refresh=${Date.now()}`);if(app!==resultApp||states[app].job?.id!==job.id)return;job.result=latest;}catch(error){if(app!==resultApp)return;}
  const url=name=>`/files/${job.id}/${encodeURIComponent(name)}?timing=${job.result?.video_timing_version||0}`;
  const choices=app==='wapr'?[['overlay.png','位姿叠加']]:[['tracks.mp4','3D 轨迹'],['se3.mp4','SE(3) 运动'],['rigidity.mp4','刚性分组'],['query_points.png','查询点']];
  $('result-tabs').replaceChildren();const show=(name,button)=>{document.querySelectorAll('#result-tabs button').forEach(b=>b.classList.toggle('active',b===button));const media=document.createElement(name.endsWith('.mp4')?'video':'img');media.src=url(name);if(media.tagName==='VIDEO'){media.controls=true;media.playsInline=true;media.preload='metadata';}else{media.alt='模型输出可视化';media.className='result-image';}$('result-media').replaceChildren(media);};
  for(const [name,label] of choices.filter(([name])=>job.artifacts.includes(name))){const button=document.createElement('button');button.type='button';button.textContent=label;button.onclick=()=>show(name,button);$('result-tabs').append(button);}
  $('result-tabs').firstChild?.click();$('result-json').textContent=JSON.stringify(job.result,null,2);
  const r=job.result||{};const entries=app==='wapr'?[['排序分数',Number(r.score_6d).toFixed(4)],['耗时（含初始化）',`${r.elapsed_seconds_including_setup} s`],['坐标单位','米']]:[['采样帧数',r.frames],['推理分辨率',`${r.width} × ${r.height}`],['有效 SE(3) 比例',`${(100*r.valid_se3_fraction).toFixed(1)}%`],...(r.output_duration_seconds?[['播放时长',`${r.output_duration_seconds.toFixed(3)} s`]]:[]),...(r.export_seconds!=null?[['结果导出',`${r.export_seconds.toFixed(1)} s`]]:[]) ];
  $('metrics').replaceChildren(...entries.map(([label,value])=>{const el=document.createElement('div');el.textContent=label;el.append(Object.assign(document.createElement('strong'),{textContent:String(value)}));return el;}));
  $('downloads').replaceChildren(...job.artifacts.filter(n=>n!=='run.log').map(name=>{const a=document.createElement('a');a.href=url(name);a.download=name;a.textContent=name==='results.zip'?'↓ 下载全部结果':`↓ ${name}`;if(name==='results.zip')a.title='首次下载时打包，后续下载使用缓存';return a;}));
}
$('form').addEventListener('submit',async e=>{
  e.preventDefault();const jobApp=app, required=app==='wapr'?['rgb','depth','mesh',...(values().region==='mask'?['mask']:[])]:['video'];
  if(required.some(k=>!states[app].files[k])){message(app==='wapr'?'请上传 RGB、深度、网格，以及所选模式需要的 Mask。':'请先上传视频。');return;}
  const body={app,...values()};if(app==='mose3'&&!body.sample_all&&!body.sampling_fps&&body.query_idx>=body.num_frames){message('查询帧索引必须小于采样帧数。');return;}
  if(app==='mose3'&&body.point_mode==='manual'){
    const p=states.mose3.picker;if(!p||p.key!==samplingKey()){message('请先生成帧预览，再点击选择查询点。');return;}
    if(!p.points.length){message('请在查询帧上至少点击一个点。');return;}
    body.query_points=p.points.map(point=>[...point]);body.point_preview_id=p.preview.id;body.query_idx=p.frame;
  }
  message('');const abort=new AbortController();uploadAbort=abort;renderJob(states[app].job);
  try{
    for(const key of required){const file=states[jobApp].files[key];$('status').textContent=`上传 ${file.name}…`;let token=uploadCache.get(file);if(!token){const result=await api('/api/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name),'Content-Type':'application/octet-stream'},body:file,signal:abort.signal});token=result.token;uploadCache.set(file,token);}body[key]=token;}
    let job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(abort.signal.aborted)job=await api(`/api/jobs/${job.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    states[jobApp].job=job;localStorage.setItem(`pose-lab-${jobApp}`,job.id);if(app===jobApp){renderJob(job);switchView('log');}poll(jobApp,job.id);
  }catch(error){if(error.name!=='AbortError')message(error.message);else message('已取消上传。');}
  finally{uploadAbort=null;if(app===jobApp)renderJob(states[jobApp].job);}
});
async function poll(jobApp,id){try{const job=await api(`/api/jobs/${id}`);if(states[jobApp].job&&states[jobApp].job.id!==id)return;states[jobApp].job=job;if(app===jobApp){renderJob(job);if(job.status==='succeeded')switchView('result');if(job.status==='failed')message('推理失败，请查看运行日志中的具体原因。');}if(['running','queued'].includes(job.status))setTimeout(()=>poll(jobApp,id),1200);}catch(error){if(app===jobApp)message(`无法获取任务状态：${error.message}。刷新页面可重新连接。`);}}
$('cancel').addEventListener('click',async()=>{if(uploadAbort){uploadAbort.abort();return;}const job=states[app].job;if(!job)return;try{states[app].job=await api(`/api/jobs/${job.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});renderJob(states[app].job);}catch(error){message(error.message);}});
$('reset').addEventListener('click',()=>{states[app]={files:{},values:{},job:null};localStorage.removeItem(`pose-lab-${app}`);$('input-fields').replaceChildren();render(app);});
$('example').addEventListener('click',async()=>{
  const selectedApp=app;$('example').disabled=true;message('正在载入官方样例…',true);
  try{
    const sample=await api(`/api/example/${selectedApp}`);
    const files=await Promise.all(Object.entries(sample.files).map(async([key,info])=>{const response=await fetch(info.url);if(!response.ok)throw Error('无法读取样例文件');return [key,new File([await response.blob()],info.name)];}));
    states[selectedApp].files=Object.fromEntries(files);states[selectedApp].values={...states[selectedApp].values,...sample.values};if(selectedApp==='mose3'){states.mose3.picker=null;queryImage=null;}
    if(app===selectedApp){$('input-fields').replaceChildren();render(app);message('官方样例已载入，可直接开始推理。',true);}
  }catch(error){if(app===selectedApp)message(error.message);}
  finally{if(app===selectedApp)$('example').disabled=false;}
});
document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>switchView(el.dataset.view)));
document.querySelectorAll('[data-app]').forEach(el=>el.addEventListener('click',e=>{e.preventDefault();if(uploadAbort)return;history.pushState({},'',el.getAttribute('href'));render(el.dataset.app);}));
window.addEventListener('popstate',()=>render(location.pathname==='/mose3'?'mose3':'wapr'));
(async()=>{try{config=await api('/api/config');render(location.pathname==='/mose3'?'mose3':config.app==='mose3'?'mose3':'wapr');for(const key of ['wapr','mose3']){if(config.app!=='all'&&config.app!==key)continue;const id=localStorage.getItem(`pose-lab-${key}`);if(id&&/^[a-f0-9]{32}$/.test(id))poll(key,id);}}catch(error){message(`无法连接本地服务：${error.message}`);$('run').disabled=true;}})();
