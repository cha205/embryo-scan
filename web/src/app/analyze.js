// Analyze tab: a real day-5 photo on a glass slide under the objective, with
// the grading model's attention (Grad-CAM) rising off it as a 3D relief.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { API } from '../shared/api.js'
import '../shared/small-screen.js'

const W = 2.4
const H = 1.8
const samples = await fetch('/blastocyst/samples.json').then((r) => r.json())

const EXPLAIN = {
  exp: ['Early blastocyst: the cavity is less than half the embryo.', 'Blastocyst: the cavity is half the embryo or more.',
    'Full blastocyst: the cavity fills the embryo.', 'Expanded: the cavity has grown and the shell is thinning.',
    'Hatching: the embryo is starting to break out of its shell.'],
  icm: ['many cells, tightly packed (best)', 'several cells, loosely grouped', 'very few cells'],
  te: ['many cells forming an even layer (best)', 'few cells, a looser layer', 'very few, large cells'],
}

// ---------------------------------------------------------------- scene
const canvas = document.querySelector('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
renderer.toneMapping = THREE.NeutralToneMapping
const scene = new THREE.Scene()
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture
scene.add(new THREE.Mesh(new THREE.SphereGeometry(60, 48, 24), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false,
  uniforms: { uTop: { value: new THREE.Color('#f4f6f9') }, uBottom: { value: new THREE.Color('#cfd6df') } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `uniform vec3 uTop, uBottom; varying vec3 vP;
    void main(){ gl_FragColor = vec4(mix(uBottom, uTop, smoothstep(-0.3, 0.5, vP.y)), 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    }`,
})))
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200)
camera.position.set(0.0, 3.6, 4.4)
const controls = new OrbitControls(camera, canvas)
controls.target.set(0, 0.15, 0)
controls.enableDamping = true
controls.enablePan = false
controls.minDistance = 2.6
controls.maxDistance = 9
controls.maxPolarAngle = 1.35
const sun = new THREE.DirectionalLight('#ffffff', 1.6)
sun.position.set(2, 5, 3)
scene.add(sun)

// microscope stage: a metal plate with a glowing aperture under the slide
const stage = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.12, 3.8), new THREE.MeshStandardMaterial({ color: '#d3d8df', metalness: 0.75, roughness: 0.42 }))
stage.position.y = -0.1
scene.add(stage)
{
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  grad.addColorStop(0, 'rgba(255,255,255,1)')
  grad.addColorStop(0.6, 'rgba(235,242,255,0.8)')
  grad.addColorStop(1, 'rgba(235,242,255,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, 128, 128)
  const glow = new THREE.Mesh(new THREE.CircleGeometry(1.5, 96), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }))
  glow.rotation.x = -Math.PI / 2
  glow.position.y = -0.035
  scene.add(glow)
}
const slideGlass = new THREE.Mesh(new THREE.BoxGeometry(W + 0.5, 0.03, H + 0.3), new THREE.MeshPhysicalMaterial({
  color: '#eef3f8', roughness: 0.05, transparent: true, opacity: 0.35, clearcoat: 1, depthWrite: false,
}))
slideGlass.position.y = 0.0
scene.add(slideGlass)

// objective above
const objective = new THREE.Group()
const metal = (color, roughness = 0.28) => new THREE.MeshStandardMaterial({ color, metalness: 1, roughness })
const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 1.4, 96), metal('#3a404b', 0.35))
barrel.position.y = 0.7
const band = new THREE.Mesh(new THREE.CylinderGeometry(0.506, 0.506, 0.07, 96), new THREE.MeshStandardMaterial({ color: '#2f5bff', metalness: 0.3, roughness: 0.3 }))
band.position.y = 0.28
const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.33, 0.47, 0.34, 96), metal('#d5d9df', 0.14))
nose.position.y = -0.17
objective.add(barrel, band, nose)
objective.position.set(0.0, 1.9, -0.45)
scene.add(objective)

const loader = new THREE.TextureLoader()
const photo = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ color: '#d9d9d9', toneMapped: false }))
photo.rotation.x = -Math.PI / 2
photo.position.y = 0.02
scene.add(photo)

const terrainMat = new THREE.ShaderMaterial({
  transparent: true, side: THREE.DoubleSide, depthWrite: true,
  uniforms: { uHeat: { value: null }, uNext: { value: null }, uMix: { value: 0 }, uRise: { value: 0 }, uHeight: { value: 0.75 } },
  vertexShader: /* glsl */`
    uniform sampler2D uHeat, uNext;
    uniform float uMix, uRise, uHeight;
    varying float vH; varying vec2 vUv; varying vec3 vW;
    void main() {
      vUv = uv;
      float h = mix(texture2D(uHeat, uv).r, texture2D(uNext, uv).r, uMix);
      vH = h;
      vec3 p = position;
      p.z += pow(h, 1.7) * uHeight * uRise;
      vec4 w = modelMatrix * vec4(p, 1.0);
      vW = w.xyz;
      gl_Position = projectionMatrix * viewMatrix * w;
    }`,
  fragmentShader: /* glsl */`
    uniform float uRise;
    varying float vH; varying vec2 vUv; varying vec3 vW;
    vec3 ramp(float t) {
      vec3 a = vec3(0.184, 0.357, 1.0), b = vec3(0.545, 0.361, 0.965), c = vec3(1.0, 0.478, 0.271), d = vec3(1.0, 0.761, 0.29);
      return t < 0.33 ? mix(a, b, t / 0.33) : t < 0.66 ? mix(b, c, (t - 0.33) / 0.33) : mix(c, d, (t - 0.66) / 0.34);
    }
    void main() {
      vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
      float shade = 0.65 + 0.35 * abs(dot(n, normalize(vec3(0.4, 1.0, 0.5))));
      vec3 col = ramp(clamp((vH - 0.25) / 0.75, 0.0, 1.0)) * shade;
      float f = fract(vH * 10.0);
      float line = 1.0 - smoothstep(0.0, fwidth(vH * 10.0) * 1.3, min(f, 1.0 - f));
      col = mix(col, vec3(1.0), line * 0.55);
      float a = smoothstep(0.25, 0.5, vH) * uRise;
      gl_FragColor = vec4(col, a * (0.82 + 0.18 * line));
      #include <colorspace_fragment>
    }`,
})
const terrain = new THREE.Mesh(new THREE.PlaneGeometry(W, H, 192, 144), terrainMat)
terrain.rotation.x = -Math.PI / 2
terrain.position.y = 0.03
scene.add(terrain)

// ---------------------------------------------------------------- UI
const $ = (s) => document.querySelector(s)
const state = { i: -1, head: 'icm', rise: 0, riseTarget: 0, mix: null, pending: null }
const items = [] // test-set samples, then the visitor's own photos
const heatCache = {}
const heat = (url) => {
  if (!heatCache[url]) {
    const t = loader.load(url)
    t.colorSpace = THREE.NoColorSpace
    heatCache[url] = t
  }
  return heatCache[url]
}

function addItem(item, list) {
  const li = document.createElement('li')
  li.innerHTML = `<img alt=""><div><b></b><span></span></div>`
  li.querySelector('img').src = item.photo
  li.querySelector('b').textContent = item.title
  li.querySelector('span').textContent = item.expert ? `panel ${item.expert} · AI ${item.model}` : `AI ${item.model}`
  const i = items.push({ ...item, li }) - 1
  li.addEventListener('click', () => select(i))
  list.append(li)
  return i
}

samples.forEach((s, i) => addItem({ ...s, title: `Embryo ${String.fromCharCode(65 + i)}` }, $('#samples')))

function letter(k, c) { return k === 'exp' ? String(c + 1) : 'ABC'[c] }

function showResult(s) {
  $('#r-grade').textContent = s.model
  $('#r-panel').textContent = s.expert ?? ''
  $('.vs').hidden = !s.expert
  const top = (k) => s.probs[k].indexOf(Math.max(...s.probs[k]))
  const e = top('exp')
  $('#r-explain').textContent = e >= 2
    ? `${EXPLAIN.exp[e]} Inner cell mass ${letter('icm', top('icm'))}: ${EXPLAIN.icm[top('icm')]}. Trophectoderm ${letter('te', top('te'))}: ${EXPLAIN.te[top('te')]}.`
    : `${EXPLAIN.exp[e]} Inner cell mass and trophectoderm aren't graded this early.`
  const names = { exp: 'Expansion', icm: 'Inner cell mass', te: 'Trophectoderm' }
  $('#r-heads').innerHTML = Object.entries(s.probs).map(([k, p]) => {
    const t = top(k)
    const bars = p.map((v, c) => `<span class="${c === t ? 'top' : ''}" style="height:${Math.max(6, v * 100)}%"><em>${letter(k, c)}</em></span>`).join('')
    return `<div class="head-row"><div class="top"><span>${names[k]}</span><b>${letter(k, t)} · ${Math.round(p[t] * 100)}%</b></div><div class="probs">${bars}</div></div>`
  }).join('')
}

function select(i) {
  if (i === state.i) return
  items.forEach((item, j) => item.li.classList.toggle('on', j === i))
  state.pending = i
  state.riseTarget = 0 // sink the terrain, swap, then rise again
}

function apply(i) {
  state.i = i
  const s = items[i]
  const tex = loader.load(s.photo)
  tex.colorSpace = THREE.SRGBColorSpace
  photo.material.map = tex
  photo.material.needsUpdate = true
  terrainMat.uniforms.uHeat.value = heat(s.heatmaps[state.head])
  terrainMat.uniforms.uNext.value = terrainMat.uniforms.uHeat.value
  terrainMat.uniforms.uMix.value = 0
  showResult(s)
  state.riseTarget = 1
}

document.querySelectorAll('#r-seg button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.head === state.head || state.i < 0) return
  document.querySelectorAll('#r-seg button').forEach((x) => x.classList.toggle('on', x === b))
  state.head = b.dataset.head
  terrainMat.uniforms.uNext.value = heat(items[state.i].heatmaps[state.head])
  state.mix = 0
}))

// ---------------------------------------------------------------- your photo
// The grading server returns the photo as the model saw it, the grade and a heat
// map per head, in the same shape as a sample. The hosted server sleeps when
// idle, so the page keeps polling while it wakes up.
const upload = $('#upload')
const input = upload.querySelector('input')
const note = $('#upload-note')
const NOTE = note.textContent
const OFFLINE = API
  ? 'The photo grader is offline right now. Try again later.'
  : 'Grading your own photo needs the local server: uvicorn server.main:app --port 8000, then reload.'
const WAKE_TRIES = 24 // ~5 s apart: a sleeping Space takes about a minute to start
let views = 8
let uploads = 0

function setUpload(mode, message = NOTE) {
  upload.classList.toggle('busy', mode === 'busy' || mode === 'waking')
  upload.classList.toggle('offline', mode === 'offline')
  input.disabled = mode !== 'ready'
  const [icon, title, sub] = {
    busy: ['ti-loader-2', 'Grading…', `5 models × ${views} views`],
    waking: ['ti-loader-2', 'Waking the grader…', 'about a minute'],
    offline: ['ti-plug-connected-x', 'Grader offline', 'see below'],
  }[mode] ?? ['ti-upload', 'Grade a photo', 'PNG or JPG, or drop it here']
  upload.querySelector('.ico i').className = `ti ${icon}`
  $('#upload-title').textContent = title
  $('#upload-sub').textContent = sub
  note.textContent = mode === 'offline' ? OFFLINE : message
  note.classList.toggle('error', message !== NOTE)
}

async function connect() {
  const tries = API ? WAKE_TRIES : 1
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(10000) })
      const h = r.ok ? await r.json() : null
      if (h?.models) {
        views = h.views ?? 8
        setUpload('ready')
        return
      }
    } catch {
      // asleep, still starting, or not running
    }
    if (i + 1 < tries) {
      setUpload('waking')
      await new Promise((resolve) => setTimeout(resolve, 5000))
    }
  }
  setUpload('offline')
}

async function gradeFile(file) {
  if (!file || input.disabled) return
  if (!['image/png', 'image/jpeg'].includes(file.type)) {
    setUpload('ready', 'Use a PNG or JPG photo.')
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    setUpload('ready', 'That image is larger than 10 MB.')
    return
  }
  setUpload('busy')
  try {
    const body = new FormData()
    body.append('file', file)
    const r = await fetch(`${API}/api/grade`, { method: 'POST', body })
    const res = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(res.detail ?? OFFLINE)
    uploads += 1
    const i = addItem({ photo: res.photo, heatmaps: res.heatmaps, probs: res.probs, model: res.grade, expert: null, title: `Your photo ${uploads}` }, $('#uploads'))
    setUpload('ready')
    select(i)
  } catch (err) {
    setUpload('ready', err instanceof TypeError ? OFFLINE : err.message)
  }
}

input.addEventListener('change', () => {
  gradeFile(input.files[0])
  input.value = '' // allow choosing the same file again
})
upload.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !input.disabled) {
    e.preventDefault()
    input.click()
  }
})
window.addEventListener('dragover', (e) => {
  e.preventDefault()
  document.body.classList.toggle('dropping', !input.disabled)
})
window.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('dropping') })
window.addEventListener('drop', (e) => {
  e.preventDefault()
  document.body.classList.remove('dropping')
  gradeFile(e.dataTransfer.files[0])
})
connect()

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const clock = new THREE.Clock()
let last = 0
select(0)

renderer.setAnimationLoop(() => {
  const now = clock.getElapsedTime()
  const dt = Math.min(1, now - last)
  last = now
  state.rise += Math.sign(state.riseTarget - state.rise) * Math.min(Math.abs(state.riseTarget - state.rise), dt * 1.4)
  if (state.pending !== null && state.rise <= 0.001) { apply(state.pending); state.pending = null }
  terrainMat.uniforms.uRise.value = ease(state.rise)
  if (state.mix !== null) {
    state.mix = Math.min(1, state.mix + dt / 0.7)
    terrainMat.uniforms.uMix.value = ease(state.mix)
    if (state.mix >= 1) {
      terrainMat.uniforms.uHeat.value = terrainMat.uniforms.uNext.value
      terrainMat.uniforms.uMix.value = 0
      state.mix = null
    }
  }
  objective.position.y = 1.9 + 0.02 * Math.sin(now * 0.6)
  controls.update()
  renderer.render(scene, camera)
})
