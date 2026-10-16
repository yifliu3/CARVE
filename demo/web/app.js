import {ProxyRenderer} from './render.js';
import {Recording} from './recording.js';

const $ = id => document.getElementById(id);
const renderer = new ProxyRenderer($('canvas'));
const recording = new Recording([0,0,0,0,0]);
const keys = new Set();
let scene = null, health = null, busy = false, committed = [0,0,0,0,0];
let scenes = [];
let replayIndex = 0, replayTime = 0;

async function api(url, options) {
 const response = await fetch(url, options);
 if (!response.ok) {
  const text = await response.text();
  try { throw Error(JSON.parse(text).error || text); }
  catch (error) { if (error instanceof SyntaxError) throw Error(text); throw error; }
 }
 return response.json();
}
function report(error) {
 $('status').textContent = error.message;
 $('status').classList.add('error');
}
function sync() {
 $('frames').textContent = `${recording.poses.length} frames / ${(recording.poses.length/24).toFixed(2)} s`;
 $('mode').textContent = busy ? 'Processing' : recording.replay ? 'Replay' : recording.active ? 'Recording' : 'Idle';
 $('record').textContent = recording.active ? '\u23f8' : '\u25cf';
 $('encode').disabled = busy || !health?.encoding_available || !recording.poses.length;
 $('prepare').disabled = busy || !health?.reference_preparation_available || !scene?.reference_available || scene?.static_ready || !!health?.active_scene;
 $('generate').disabled = busy || !health || health.preview || !scene || scene.preview_only || !(scene.static_ready || health.configured_static) || !recording.poses.length;
 for (const id of ['record','replay','clear','import','scenes']) $(id).disabled = busy;
}
async function refresh() {
 health = await api('/api/health');
 $('status').classList.remove('error');
 $('status').textContent = health.preview ? 'Preview only' : `v4 ${health.stage === 'causal_dmd_sgf' ? 'SGF' : 'AnyFlow'} / ${health.weights.toUpperCase()}`;
 sync();
}
async function load(id) {
 if (busy) return;
 if (health?.active_scene && health.active_scene !== id) {
  await api('/api/stop', {method:'POST'});
  await refresh();
 }
 await renderer.load(`/scenes/${id}/environment`);
 scene = {id, ...renderer.meta, ...scenes.find(item => item.id === id)};
 committed = health?.active_scene === id ? [...health.committed_pose] : [0,0,0,0,0];
 recording.reset(committed);
 $('scene-name').textContent = scene.name;
 $('scene-kind').textContent = scene.preview_only ? 'Preview only' : scene.static_ready ? 'Reference ready' : 'Imported mesh';
 renderer.render(recording.pose);
 sync();
}
async function list(selectId) {
 const data = await api('/api/scenes');
 scenes = data.scenes;
 $('scenes').replaceChildren(...data.scenes.map(item => {
  const option = document.createElement('option');
  option.value = item.id; option.textContent = item.name;
  return option;
 }));
 $('scenes').value = selectId || 'calibration';
 await load($('scenes').value);
}
$('scenes').onchange = () => load($('scenes').value).catch(report);
$('speed').oninput = () => $('speed-value').textContent = `${$('speed').value} m/s`;
$('canvas').onkeydown = event => {
 if (!['w','a','s','d','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
 event.preventDefault();
 if (busy) return;
 keys.add(event.key);
 if (!recording.active && !recording.replay) recording.begin();
};
$('canvas').onkeyup = event => keys.delete(event.key);
$('canvas').onblur = () => { keys.clear(); recording.pause(); sync(); };
document.addEventListener('visibilitychange', () => {
 if (document.hidden) {keys.clear(); recording.pause(); sync();}
});
$('record').onclick = () => {
 recording.active ? recording.pause() : recording.begin();
 $('canvas').focus(); sync();
};
$('clear').onclick = () => { recording.reset(committed); keys.clear(); sync(); };
$('replay').onclick = () => {
 if (!recording.poses.length) return;
 recording.pause(); recording.replay = true; replayIndex = 0; replayTime = 0; sync();
};
function download(name, data) {
 const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}));
 const anchor = document.createElement('a');
 anchor.href = url; anchor.download = name; anchor.click();
 setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('export').onclick = () => download('trajectory.json', recording.snapshot());
$('import').onchange = async event => {
 try {
  const file = event.target.files[0]; if (!file) return;
  const value = JSON.parse(await file.text());
  if (value.fps !== 24 || !Array.isArray(value.poses) || !value.poses.length || value.poses.length > 720) throw Error('Invalid 24 fps trajectory');
  for (const pose of [...value.poses, value.end_pose]) {
   if (!Array.isArray(pose) || pose.length !== 5 || !pose.every(Number.isFinite) || pose.slice(0,3).some(n=>Math.abs(n)>100) || Math.abs(pose[4])>Math.PI*.44) throw Error('Invalid camera pose');
  }
  if (value.poses[0].some((n,i)=>Math.abs(n-committed[i])>1e-5)) throw Error('Trajectory must start at the current model pose');
  for(let i=0;i<value.poses.length;i++){
   const p=value.poses[i], q=value.poses[i+1]||value.end_pose;
   const yaw=((q[3]-p[3]+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;
   if(Math.hypot(...p.slice(0,3).map((n,j)=>q[j]-n))>.6/24+1e-5 || Math.max(Math.abs(yaw),Math.abs(q[4]-p[4]))>Math.PI/144+1e-5)throw Error('Trajectory exceeds motion limits');
  }
  recording.reset(committed); recording.poses=value.poses.map(p=>[...p]); recording.pose=[...value.end_pose]; sync();
 } catch(error) {report(error);}
 event.target.value='';
};
$('upload').onsubmit = async event => {
 event.preventDefault(); if (busy) return;
 busy=true; sync();
 try {
  const form = new FormData(event.target);
  for(const [key,value] of [...form.entries()]) if(value instanceof File && !value.size) form.delete(key);
  const result=await api('/api/assets', {method:'POST', body:form});
  busy=false; await list(result.id);
 } catch(error) {report(error);}
 finally {busy=false; sync();}
};
$('prepare').onclick = async () => {
 recording.pause(); keys.clear(); busy=true; sync();
 try {
  await api(`/api/scenes/${scene.id}/prepare-static`, {method:'POST'});
  scene.static_ready=true;
  $('scene-kind').textContent='Reference ready';
  await refresh();
 } catch(error) {report(error);}
 finally {busy=false;sync();}
};
async function process(operation) {
 recording.pause(); recording.replay=false; keys.clear(); busy=true; sync();
 try {
  const result=await api(`/api/scenes/${scene.id}/process`, {
   method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({operation,trajectory:recording.snapshot()})
  });
  if(operation==='generate'){committed=[...result.pose];recording.reset(committed);}
  const row=document.createElement('div'); row.className='result';
  const title=document.createElement('span'); title.textContent=`${operation} / ${result.frames} frames`;row.append(title);
  if(result.video){
   const video=document.createElement('video');video.src=result.video;video.controls=true;video.preload='metadata';row.append(video);
   const link=document.createElement('a');link.href=result.video;link.download='generated.mp4';link.textContent='MP4';row.append(link);
  }
  result.blocks.forEach((block,index)=>{
   for(const kind of ['tokens','latents']) if(block[kind]){
    const link=document.createElement('a');link.href=block[kind];link.textContent=`${kind} ${index+1}`;row.append(link);
   }
  });
  $('results').prepend(row);
  await refresh();
 } catch(error) {
  if(operation==='generate'){
   try {
    health=await api('/api/health');
    committed=[...health.committed_pose];
    recording.reset(committed);
   } catch {}
  }
  report(error);
 }
 finally {busy=false;sync();}
}
$('encode').onclick=()=>process('encode');
$('generate').onclick=()=>process('generate');
$('stop').onclick=async()=>{
 try {await api('/api/stop',{method:'POST'});committed=[0,0,0,0,0];recording.reset(committed);await refresh();}
 catch(error){report(error);}
};
let previous=performance.now();
function animate(now){
 const dt=Math.min(.1,(now-previous)/1000);previous=now;
 const axis=(positive,negative)=>Number(keys.has(positive))-Number(keys.has(negative));
 if(!busy)recording.advance(dt,{forward:axis('w','s'),right:axis('d','a'),yaw:axis('ArrowRight','ArrowLeft'),pitch:axis('ArrowUp','ArrowDown')},Number($('speed').value),30);
 let pose=recording.pose;
 if(recording.replay){
  replayTime+=dt;replayIndex=Math.min(recording.poses.length-1,Math.floor(replayTime*24));
  pose=recording.poses[replayIndex];
  if(replayTime>=recording.poses.length/24)recording.replay=false;
 }
 if(scene)renderer.render(pose);
 $('pose').textContent=`X ${pose[0].toFixed(2)} / Y ${pose[1].toFixed(2)} / Z ${pose[2].toFixed(2)}`;
 sync();requestAnimationFrame(animate);
}
window.demo={renderer,recording};
refresh().then(()=>list()).then(()=>requestAnimationFrame(animate)).catch(report);
