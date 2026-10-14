"""Small offline Canvas playback of saved molecular chemotaxis trajectories."""

# Inline HTML, CSS and JavaScript keep the offline renderer self-contained.
# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _lane(path: Path, stride: int, config: dict, label: str, genome) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "position" not in data or "initial_position" not in data:
            raise ValueError("animation requires saved chemotaxis positions")
        position = data["position"]
        if position.ndim != 4 or position.shape[-1] != 2 or not all(position.shape):
            raise ValueError("position must have shape (time, replicate, episode, 2)")
        horizon = len(position)
        steps = list(range(0, horizon + 1, stride))
        if steps[-1] != horizon:
            steps.append(horizon)
        if len(steps) > 10000:
            raise ValueError("animation exceeds 10000 frames; increase frame_stride")
        initial = data["initial_position"]
        if initial.shape != (1, *position.shape[1:]):
            raise ValueError("initial_position shape does not match position")
        indices = np.asarray(steps[1:]) - 1

        def channel(name, *, initial_name=None):
            if name not in data:
                return None
            value = data[name]
            if value.shape[:3] != position.shape[:3]:
                raise ValueError(f"{name} shape does not match position")
            sampled = value[indices, 0, 0]
            start = data[initial_name][0, 0, 0] if initial_name in data else sampled[0]
            return np.concatenate((np.asarray(start)[None], sampled)).tolist()

        lane: dict[str, Any] = dict(
            label=label,
            genome=genome,
            steps=steps,
            position=np.concatenate((initial[:, 0, 0], position[indices, 0, 0])).tolist(),
            theta=channel("theta", initial_name="initial_theta"),
            orientation=channel("orientation", initial_name="initial_orientation"),
            tumble_remaining=channel("tumble_remaining", initial_name="initial_tumble_remaining"),
            phases=channel("phases", initial_name="initial_phases"),
            source=channel("source", initial_name="initial_source"),
            exposure=channel("sensed_exposure"),
            tumble=channel("tumbling"),
            cues={},
            readouts={},
        )
        for name in data.files:
            if name.startswith("input_"):
                lane["cues"][name[6:]] = channel(name)
            if name.startswith("readout_"):
                lane["readouts"][name[8:]] = channel(name)
        if config["field"] == "fluctuating" and lane["phases"] is None:
            raise ValueError("fluctuating animation requires saved phases")
        if config["field"] == "source" and lane["source"] is None:
            raise ValueError("source animation requires saved source positions")
        return lane


def render_molecular_animation(root: str | Path, *, frame_stride: int = 5) -> Path:
    """Write animation.html; show replicate 1, episode 1 with sampled saved states.

    Saved post-step phases reconstruct each field. Cue, sensed concentration and
    target occupancy describe the decision that led to the displayed position.
    The initial frame has no decision telemetry. No simulation is advanced.
    """
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    root = Path(root)
    summary = json.loads((root / "experiment.json").read_text())
    task = summary["task"]
    if task["kind"] != "chemotaxis":
        raise ValueError("animation requires a chemotaxis experiment")
    config = task["config"]
    final_genomes = summary.get("selected_genomes", [])
    lanes = [
        _lane(
            root / "behaviour.npz",
            frame_stride,
            config,
            "Final selected cell",
            final_genomes[0] if final_genomes else None,
        )
    ]
    initial_path = root / "initial_behaviour.npz"
    if initial_path.exists():
        genomes = summary.get("initial_selected_genomes", [])
        lanes.insert(
            0,
            _lane(
                initial_path,
                frame_stride,
                config,
                "Initial selected cell",
                genomes[0] if genomes else None,
            ),
        )
        if lanes[0]["steps"] != lanes[1]["steps"]:
            raise ValueError("initial and final animation horizons must match")
        for name in ("position", "phases", "source"):
            if lanes[0][name] is not None and lanes[1][name] is not None:
                if not np.allclose(lanes[0][name][0], lanes[1][name][0]):
                    raise ValueError("initial and final animation scenarios must match")
    payload = dict(config=config, dt=summary["episode"]["dt"], stride=frame_stride, lanes=lanes)
    serialized = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    output = root / "animation.html"
    output.write_text(_HTML.replace("__PAYLOAD__", serialized), encoding="utf-8")
    return output


_HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Molecular chemotaxis · playback</title>
<style>
:root{font-family:system-ui,sans-serif;color:#25372f;background:#f7f8f3}body{max-width:1080px;margin:0 auto;padding:32px 22px}h1{font-size:25px;font-weight:550;margin:0 0 8px}p{color:#617067;line-height:1.5;font-size:14px}#lanes{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}h2{font-size:16px;font-weight:550}canvas{display:block;width:100%;aspect-ratio:1;background:#e9eee3;border-radius:6px}.axis{display:flex;justify-content:space-between;color:#667368;font-size:12px;margin:7px 0}.metrics{font-size:13px;line-height:1.7;min-height:46px}.legend{display:flex;align-items:center;gap:10px;font-size:12px;margin:12px 0}.ramp{display:inline-block;width:180px;height:14px;background:linear-gradient(to right,rgb(245,246,220),rgb(95,150,140));border:1px solid #ccd4c9}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:24px 0}button,select{font:inherit;background:#fff;border:1px solid #ccd4c9;border-radius:5px;padding:7px 12px}input{accent-color:#315f4b;flex:1;min-width:150px}output{font-variant-numeric:tabular-nums;font-size:13px}.legend{display:flex;align-items:center;gap:10px;font-size:12px;color:#617067}.scale{width:120px;height:9px;background:linear-gradient(90deg,rgb(245,246,220),rgb(95,150,140));border-radius:3px}small{color:#617067}
</style>
<h1>Molecular chemotaxis</h1><p>One cell per independent world · held-out replicate 1, episode 1. Each rod follows saved motion on the periodic concentration field.</p>
<div id="lanes"></div>
<div class="controls"><button id="play" type="button">Play</button><input id="scrub" aria-label="Playback frame" type="range" min="0" value="0"><label>Speed <select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="5">5×</option><option value="0.25">¼×</option></select></label><output id="time" aria-live="off"></output></div>
<div class="legend"><span>Concentration 0</span><span class="scale"></span><span>1</span></div>
<p id="note"></p><div class="legend"><span>Concentration 0</span><span class="ramp"></span><span>1 · higher concentration →</span></div><small>Concentration is normalised. Rod and flagellum drawings are schematic. Cue, sensed concentration and occupancy describe the decision preceding the displayed position. The initial frame shows the starting state.</small>
<script id="animation-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('animation-data').textContent),cfg=data.config,L=cfg.domain;
const scrub=document.getElementById('scrub'),play=document.getElementById('play'),speed=document.getElementById('speed');
let frame=0,playing=false,last=0,accum=0;
scrub.max=data.lanes[0].steps.length-1;
document.getElementById('note').textContent=`${cfg.field==='source'?'Single exponential source; cross marks its location.':'Fourier landscape with multiple peaks; '+(cfg.static?'static':'time-varying')+'.'} The background reconstructs the simulation field from saved ${cfg.field==='source'?'source positions':'Fourier phases'} on a 192×192 display grid. Darker colour means higher concentration, using a fixed 0–1 scale. ${cfg.motion_model==='ecoli_spherical_v1'?'3D orientation projected onto this concentration plane; finite tumbles with active rotational diffusion; time in seconds, '+cfg.micrometers_per_length_unit+' µm per position unit.':'Instantaneous heading-reset tumbles; dimensionless time and position.'}`;
const views=data.lanes.map(lane=>{const section=document.createElement('section'),heading=document.createElement('h2'),canvas=document.createElement('canvas'),axis=document.createElement('div'),metrics=document.createElement('div');heading.textContent=lane.label;canvas.width=640;canvas.height=640;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`${lane.label}: cell trajectory on the concentration field`);axis.className='axis';axis.textContent=`y ↑ 0–${L}  ·  x → 0–${L} (dimensionless)`;metrics.className='metrics';section.append(heading,canvas,axis,metrics);document.getElementById('lanes').append(section);return {lane,canvas,ctx:canvas.getContext('2d'),metrics,cache:new Map()};});
function field(view,i){if(view.cache.has(i))return view.cache.get(i);const n=192,c=document.createElement('canvas');c.width=c.height=n;const ctx=c.getContext('2d'),img=ctx.createImageData(n,n);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const px=x*L/n,py=(n-1-y)*L/n;let v;if(cfg.field==='source'){const source=view.lane.source[i];let dx=px-source[0],dy=py-source[1];dx-=L*Math.floor(dx/L+.5);dy-=L*Math.floor(dy/L+.5);v=cfg.source_peak*Math.exp(-Math.hypot(dx,dy)/cfg.source_length);}else{v=cfg.mean+cfg.amplitude*cfg.modes.reduce((sum,m,j)=>sum+Math.cos(2*Math.PI*(px*m[0]+py*m[1])/L+view.lane.phases[i][j]),0)/cfg.modes.length;}const k=4*(y*n+x);img.data[k]=245-150*v;img.data[k+1]=246-96*v;img.data[k+2]=220-80*v;img.data[k+3]=255;}ctx.putImageData(img,0,0);view.cache.set(i,c);if(view.cache.size>8)view.cache.delete(view.cache.keys().next().value);return c;}
function draw(){scrub.value=frame;document.getElementById('time').textContent=`Step ${data.lanes[0].steps[frame]} · t ${(data.lanes[0].steps[frame]*data.dt).toFixed(2)}`;for(const v of views){const {lane,ctx}=v,S=640;ctx.drawImage(field(v,frame),0,0,S,S);if(cfg.field==='source'){const source=lane.source[frame],sx=source[0]/L*S,sy=S-source[1]/L*S;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx-6,sy);ctx.lineTo(sx+6,sy);ctx.moveTo(sx,sy-6);ctx.lineTo(sx,sy+6);ctx.stroke();}ctx.lineWidth=1.7;ctx.strokeStyle='rgba(30,67,51,.38)';ctx.beginPath();for(let j=0;j<=frame;j++){const p=lane.position[j],q=j?lane.position[j-1]:p;if(j===0||Math.abs(p[0]-q[0])>L/2||Math.abs(p[1]-q[1])>L/2)ctx.moveTo(p[0]/L*S,S-p[1]/L*S);else ctx.lineTo(p[0]/L*S,S-p[1]/L*S);}ctx.stroke();const p=lane.position[frame],theta=lane.theta?lane.theta[frame]:0;for(const dx of [-S,0,S])for(const dy of [-S,0,S]){ctx.save();ctx.translate(p[0]/L*S+dx,S-p[1]/L*S+dy);ctx.rotate(-theta);ctx.strokeStyle='#315e49';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(-10,0);ctx.bezierCurveTo(-18,-5,-20,5,-27,0);ctx.stroke();ctx.fillStyle='#f9fcf5';ctx.strokeStyle='#244d39';ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(-10,-4.5,20,9,4.5);ctx.fill();ctx.stroke();ctx.fillStyle='#315e49';ctx.beginPath();ctx.arc(5,0,1.5,0,7);ctx.fill();ctx.restore();}const fmt=x=>typeof x==='number'?Number(x.toFixed(3)):JSON.stringify(x);v.metrics.textContent=frame===0?'Starting state':[`Sensed before move ${fmt(lane.exposure?.[frame])}`, ...Object.entries(lane.cues).map(([k,a])=>`${k} ${fmt(a[frame])}`),...Object.entries(lane.readouts).map(([k,a])=>`${k} occupancy ${fmt(a[frame])}`),lane.tumble?.[frame]?'Tumble during this step':'Run', ...(lane.orientation?[`Out-of-plane direction ${fmt(lane.orientation[frame][2])}`]:[]), ...(lane.tumble_remaining?[`Tumble remaining ${fmt(lane.tumble_remaining[frame])} s`]:[])].join(' · ');}}
function setPlaying(value){playing=value;play.textContent=playing?'Pause':'Play';last=0;accum=0;}
play.onclick=()=>{if(frame===Number(scrub.max))frame=0;setPlaying(!playing);draw();};scrub.oninput=()=>{frame=Number(scrub.value);setPlaying(false);draw();};
function tick(now){if(playing){if(last)accum+=(now-last)*Number(speed.value)/1000;last=now;while(frame<Number(scrub.max)&&accum>=(data.lanes[0].steps[frame+1]-data.lanes[0].steps[frame])*data.dt){accum-=(data.lanes[0].steps[frame+1]-data.lanes[0].steps[frame])*data.dt;frame++;}if(frame===Number(scrub.max))setPlaying(false);draw();}requestAnimationFrame(tick);}draw();requestAnimationFrame(tick);
</script></html>"""
