// Direction C: "Attention relief" — real microscope photos laid out like slides
// on a light table. Where the AI looked rises off each photo as a glowing 3D
// terrain (Grad-CAM), and a column of light shows the success estimate.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js'

const DEMO_PROBS = [0.41, 0.24, 0.12] // placeholder until the outcome model exists
const NAMES = ['Embryo A', 'Embryo B', 'Embryo C']
const W = 2.4
const H = 1.8

const canvas = document.querySelector('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.toneMapping = THREE.ACESFilmicToneMapping
const scene = new THREE.Scene()
scene.background = new THREE.Color('#0b0c0f')
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100)
const HOME = { pos: new THREE.Vector3(0, 6.2, 9.6), target: new THREE.Vector3(0, 0.2, 0.2) }
camera.position.copy(HOME.pos)
const controls = new OrbitControls(camera, canvas)
controls.target.copy(HOME.target)
controls.enableDamping = true
controls.enablePan = false
controls.minDistance = 2.5
controls.maxDistance = 14
controls.maxPolarAngle = 1.35
scene.add(new THREE.AmbientLight('#ffffff', 0.6))
const sun = new THREE.DirectionalLight('#ffffff', 1.2)
sun.position.set(2, 6, 4)
scene.add(sun)

// Light table: faint grid fading out from the centre
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  vertexShader: 'varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `varying vec2 vP;
    void main(){
      vec2 g = abs(fract(vP * 2.0 - 0.5) - 0.5) / fwidth(vP * 2.0);
      float line = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
      float fade = 1.0 - smoothstep(2.0, 11.0, length(vP));
      gl_FragColor = vec4(vec3(0.55, 0.52, 0.5), line * fade * 0.16);
    }`,
}))
floor.rotation.x = -Math.PI / 2
floor.position.y = -0.03
scene.add(floor)

const INFERNO = /* glsl */`
  vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
  }`

const shared = { uHeight: { value: 0.6 }, uTime: { value: 0 } }

function terrainMaterial(heat) {
  return new THREE.ShaderMaterial({
    transparent: true, side: THREE.DoubleSide,
    uniforms: { ...shared, uHeat: { value: heat }, uHeatNext: { value: heat }, uMix: { value: 0 }, uRise: { value: 0 } },
    vertexShader: /* glsl */`
      uniform sampler2D uHeat, uHeatNext;
      uniform float uMix, uHeight, uRise;
      varying float vHeat; varying vec2 vUv;
      void main() {
        vUv = uv;
        float h = mix(texture2D(uHeat, uv).r, texture2D(uHeatNext, uv).r, uMix);
        vHeat = h;
        vec3 p = position;
        p.z += pow(h, 1.6) * uHeight * uRise;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */`
      uniform float uRise;
      varying float vHeat; varying vec2 vUv;
      ${INFERNO}
      void main() {
        float h = vHeat;
        vec3 col = inferno(0.22 + h * 0.72) * 0.8;
        float f = fract(h * 9.0);
        float line = 1.0 - smoothstep(0.0, fwidth(h * 9.0) * 1.4, min(f, 1.0 - f));
        vec2 g = abs(fract(vUv * vec2(64.0, 48.0) - 0.5) - 0.5) / fwidth(vUv * vec2(64.0, 48.0));
        float grid = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
        col += line * vec3(1.0, 0.8, 0.55) * 0.35 + grid * 0.05;
        float a = smoothstep(0.2, 0.5, h) * uRise;
        gl_FragColor = vec4(col, a * (0.62 + 0.3 * line));
      }`,
  })
}

function columnMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: 'varying vec2 vUv; void main(){ vec3 c = mix(vec3(0.6,0.18,0.05), vec3(1.0,0.75,0.4), vUv.y); gl_FragColor = vec4(c * (0.5 + 0.8 * vUv.y), 1.0); }',
  })
}

const composer = new EffectComposer(renderer)
composer.addPass(new RenderPass(scene, camera))
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.45, 0.4, 0.8)
composer.addPass(bloom)
composer.addPass(new OutputPass())

const loader = new THREE.TextureLoader()
const plates = []
const labelsRoot = document.querySelector('#labels')
let head = 'icm'

function buildPlate(sample, i) {
  const g = new THREE.Group()
  const off = i - 1
  g.position.set(off * 2.85, 0, Math.abs(off) * 0.45)
  g.rotation.y = -off * 0.12
  scene.add(g)

  const glass = new THREE.Mesh(new THREE.BoxGeometry(W + 0.16, 0.04, H + 0.16), new THREE.MeshPhysicalMaterial({
    color: '#d6dde6', transparent: true, opacity: 0.1, roughness: 0.15,
  }))
  glass.userData.plate = i
  g.add(glass)
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(glass.geometry), new THREE.LineBasicMaterial({ color: '#f2efe9', transparent: true, opacity: 0.3 }))
  g.add(edges)

  const photoTex = loader.load(sample.photo)
  photoTex.colorSpace = THREE.SRGBColorSpace
  const photo = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ map: photoTex, color: '#9a9a9a', toneMapped: false }))
  photo.rotation.x = -Math.PI / 2
  photo.position.y = 0.022
  g.add(photo)

  const heats = {}
  for (const [k, url] of Object.entries(sample.heatmaps)) {
    const t = loader.load(url)
    t.colorSpace = THREE.NoColorSpace
    heats[k] = t
  }
  const terrain = new THREE.Mesh(new THREE.PlaneGeometry(W, H, 192, 144), terrainMaterial(heats[head]))
  terrain.rotation.x = -Math.PI / 2
  terrain.position.y = 0.03
  g.add(terrain)

  const prob = DEMO_PROBS[i]
  const column = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 1, 24, 1, true), columnMaterial())
  const colH = prob * 3.4
  column.scale.y = colH
  column.position.set(W / 2 + 0.32, colH / 2, -H / 2 + 0.1)
  g.add(column)
  const base = new THREE.Mesh(new THREE.RingGeometry(0.08, 0.13, 48), new THREE.MeshBasicMaterial({ color: '#ff8a3d', transparent: true, opacity: 0.7, side: THREE.DoubleSide }))
  base.rotation.x = -Math.PI / 2
  base.position.set(column.position.x, 0.005, column.position.z)
  g.add(base)

  const tag = document.createElement('div')
  tag.className = `tag${i === DEMO_PROBS.indexOf(Math.max(...DEMO_PROBS)) ? ' best' : ''}`
  tag.innerHTML = `<b>${Math.round(prob * 100)}%</b><span>${NAMES[i]} · ${sample.model}</span>`
  labelsRoot.appendChild(tag)

  plates.push({ group: g, glass, terrain, heats, sample, prob, tag, top: new THREE.Vector3(column.position.x, colH + 0.25, column.position.z), lift: 0 })
}

// --- interaction -------------------------------------------------------------------
const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()
let hovered = -1
let focused = -1
const fly = { t: 1, fromPos: new THREE.Vector3(), toPos: new THREE.Vector3(), fromTarget: new THREE.Vector3(), toTarget: new THREE.Vector3() }

function flyTo(pos, target) {
  fly.t = 0
  fly.fromPos.copy(camera.position)
  fly.fromTarget.copy(controls.target)
  fly.toPos.copy(pos)
  fly.toTarget.copy(target)
}

function pick(e) {
  pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1)
  raycaster.setFromCamera(pointer, camera)
  const hit = raycaster.intersectObjects(plates.map((p) => p.glass))[0]
  return hit ? hit.object.userData.plate : -1
}
canvas.addEventListener('pointermove', (e) => {
  hovered = pick(e)
  canvas.style.cursor = hovered >= 0 ? 'pointer' : ''
})
let downAt = null
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY] })
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return
  const i = pick(e)
  if (i >= 0) focus(i)
})

const inspect = document.querySelector('#inspect')
const HEAD_NAMES = { exp: 'Expansion', icm: 'Inner cell mass', te: 'Trophectoderm' }
function focus(i) {
  focused = i
  const p = plates[i]
  const c = p.group.position
  flyTo(new THREE.Vector3(c.x + 0.35, 3.1, c.z + 2.9), new THREE.Vector3(c.x + 0.35, 0.2, c.z - 0.1))
  document.querySelector('#in-img').src = p.sample.photo
  document.querySelector('#in-name').textContent = NAMES[i]
  document.querySelector('#in-prob').textContent = `${Math.round(p.prob * 100)}%`
  document.querySelector('#in-model').textContent = p.sample.model
  document.querySelector('#in-expert').textContent = p.sample.expert
  document.querySelector('#in-bars').innerHTML = Object.entries(p.sample.probs).map(([h, probs]) => {
    const k = probs.indexOf(Math.max(...probs))
    const grade = h === 'exp' ? `grade ${k + 1}` : 'ABC'[k]
    return `<div class="bar"><span>${HEAD_NAMES[h]} ${grade}</span><span class="track"><b style="width:${probs[k] * 100}%"></b></span><span class="mono">${Math.round(probs[k] * 100)}%</span></div>`
  }).join('')
  inspect.classList.add('open')
}
document.querySelector('#btn-back').addEventListener('click', () => {
  focused = -1
  inspect.classList.remove('open')
  flyTo(HOME.pos, HOME.target)
})
document.querySelector('#height').addEventListener('input', (e) => { shared.uHeight.value = parseFloat(e.target.value) })

let swap = null
document.querySelectorAll('#heads button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.head === head) return
  document.querySelectorAll('#heads button').forEach((x) => x.classList.toggle('on', x === b))
  head = b.dataset.head
  plates.forEach((p) => { p.terrain.material.uniforms.uHeatNext.value = p.heats[head]; p.terrain.material.uniforms.uMix.value = 0 })
  swap = 0
}))

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  composer.setSize(w, h)
  bloom.resolution.set(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const clock = new THREE.Clock(false)
let last = 0
const v = new THREE.Vector3()

const samples = await fetch('/blastocyst/samples.json').then((r) => r.json())
samples.slice(0, 3).forEach(buildPlate)
clock.start()

renderer.setAnimationLoop(() => {
  const now = clock.getElapsedTime()
  const dt = Math.min(0.1, now - last)
  last = now
  shared.uTime.value = now

  if (fly.t < 1) {
    fly.t = Math.min(1, fly.t + dt / 1.1)
    const k = ease(fly.t)
    camera.position.lerpVectors(fly.fromPos, fly.toPos, k)
    controls.target.lerpVectors(fly.fromTarget, fly.toTarget, k)
  }
  if (swap !== null) {
    swap = Math.min(1, swap + dt / 0.8)
    plates.forEach((p) => {
      const u = p.terrain.material.uniforms
      u.uMix.value = ease(swap)
      if (swap >= 1) { u.uHeat.value = u.uHeatNext.value; u.uMix.value = 0 }
    })
    if (swap >= 1) swap = null
  }
  const w = window.innerWidth
  const h = window.innerHeight
  plates.forEach((p, i) => {
    const u = p.terrain.material.uniforms
    u.uRise.value = ease(THREE.MathUtils.clamp((now - 0.6 - i * 0.35) / 1.4, 0, 1))
    p.lift += ((hovered === i || focused === i ? 0.08 : 0) - p.lift) * Math.min(1, dt * 8)
    p.group.position.y = p.lift
    v.copy(p.top).applyMatrix4(p.group.matrixWorld).project(camera)
    p.tag.style.transform = `translate(${(v.x + 1) / 2 * w}px, ${(1 - v.y) / 2 * h}px) translate(-50%, -100%)`
    p.tag.style.opacity = focused === -1 || focused === i ? 1 : 0.25
  })
  controls.update()
  composer.render()
})
