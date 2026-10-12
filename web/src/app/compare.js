// Compare tab: one held-out patient's embryos on plinths. Each glass specimen
// shows that embryo at day 3, rebuilt from the divisions the AI detected in its
// video; plinth height is the predicted chance of becoming a blastocyst.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createBlastocyst } from '../shared/embryo.js'
import { wakeGrader } from '../shared/api.js'
import '../shared/small-screen.js'
import { buildTwin } from './twin.js'

wakeGrader()

const data = await fetch('/timelapse/demo.json').then((r) => (r.ok ? r.json() : null)).catch(() => null)
const cmp = data?.compare
const $ = (s) => document.querySelector(s)
if (!cmp) {
  $('#c-patient').textContent = 'No comparison data yet'
  $('#c-note').textContent = 'Run scripts/timelapse/export_demo.py after training the stage model.'
  throw new Error('no compare data')
}
const embryos = cmp.embryos
$('#c-patient').textContent = `Patient ${cmp.patient} · ${embryos.length} embryos`
$('#c-note').textContent = 'Scores come from day 0-3 timings only. Day-5 frames stay hidden until you reveal them.'

const canvas = $('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
renderer.toneMapping = THREE.NeutralToneMapping
const scene = new THREE.Scene()
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture
scene.add(new THREE.Mesh(new THREE.SphereGeometry(60, 48, 24), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false,
  uniforms: { uTop: { value: new THREE.Color('#f4f6f9') }, uBottom: { value: new THREE.Color('#cdd4de') } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `uniform vec3 uTop, uBottom; varying vec3 vP;
    void main(){ gl_FragColor = vec4(mix(uBottom, uTop, smoothstep(-0.3, 0.5, vP.y)), 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    }`,
})))
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200)
camera.position.set(0, 3.2, 6 + embryos.length * 1.95) // pull back to fit the row
const controls = new OrbitControls(camera, canvas)
controls.target.set(0, 1.25, 0)
controls.enableDamping = true
controls.enablePan = false
controls.minPolarAngle = 0.9
controls.maxPolarAngle = 1.6
const key = new THREE.DirectionalLight('#ffffff', 2)
key.position.set(3, 6, 5)
scene.add(key)
const floor = new THREE.Mesh(new THREE.CircleGeometry(14, 96), new THREE.MeshStandardMaterial({ color: '#e7ebf0', roughness: 0.9 }))
floor.rotation.x = -Math.PI / 2
scene.add(floor)

const zonaMat = new THREE.MeshPhysicalMaterial({
  color: '#ffffff', roughness: 0.04, transparent: true, opacity: 0.08, depthWrite: false,
  iridescence: 0.9, iridescenceIOR: 1.3, clearcoat: 1, envMapIntensity: 1.6,
})
zonaMat.onBeforeCompile = (shader) => {
  shader.fragmentShader = shader.fragmentShader.replace('#include <opaque_fragment>', `
    float edge = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), 2.0);
    diffuseColor.a = mix(diffuseColor.a, 0.85, edge);
    #include <opaque_fragment>`)
}
const cellMat = new THREE.MeshPhysicalMaterial({ color: '#eef2f8', roughness: 0.25, clearcoat: 0.8, sheen: 0.5, sheenColor: '#d6e2ff', iridescence: 0.2 })
const plinthMat = new THREE.MeshPhysicalMaterial({ color: '#f8f9fb', roughness: 0.5, clearcoat: 0.4 })
const accentMat = new THREE.MeshBasicMaterial({ color: '#2f5bff' })

const SPACING = 1.75
const specimens = embryos.map((e, i) => {
  const x = (i - (embryos.length - 1) / 2) * SPACING
  const h = 0.2 + e.score * 2.0 // height is proportional to the predicted chance
  const g = new THREE.Group()
  g.position.set(x, 0, -Math.abs(i - (embryos.length - 1) / 2) * 0.35)
  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.66, h, 96), plinthMat)
  plinth.position.y = h / 2
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.623, 0.006, 8, 160), i === 0 ? accentMat : new THREE.MeshBasicMaterial({ color: '#9aa3b2' }))
  ring.rotation.x = Math.PI / 2
  ring.position.y = h
  g.add(plinth, ring)

  // the embryo at day 3, from the AI-detected division times
  const ev = Object.fromEntries(Object.entries(e.events_ai).map(([k, v]) => [k, v ?? 999]))
  const twin = buildTwin(ev, 11 + i)
  const spec = new THREE.Group()
  spec.position.y = h + 0.62
  spec.scale.setScalar(0.55)
  spec.add(new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), zonaMat))
  const cells = twin.at(72)
  const inst = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 40, 28), cellMat, cells.length)
  const m = new THREE.Matrix4()
  cells.forEach((c, j) => inst.setMatrixAt(j, m.compose(c.pos, new THREE.Quaternion(), new THREE.Vector3(c.r, c.r, c.r))))
  spec.add(inst)
  const blast = createBlastocyst({
    expansion: 4, icm: 'A', te: 'A', seed: 3 + i, teSize: 0.92, teFlat: 0.03,
    materials: { zona: new THREE.MeshBasicMaterial({ visible: false }), te: cellMat, icm: new THREE.MeshPhysicalMaterial({ color: '#5277ff', clearcoat: 0.8 }) },
  })
  blast.visible = false
  spec.add(blast)
  g.add(spec)
  scene.add(g)

  const tag = document.createElement('div')
  tag.className = `tag${i === 0 ? ' first' : ''}`
  const times = ['t2', 't3', 't4', 't5', 't8'].map((k) => `${k} ${e.events_ai[k] == null ? 'not reached' : e.events_ai[k].toFixed(1) + ' h'}`).join(', ')
  tag.title = `AI-detected divisions: ${times}. Meseguer category ${e.category}.`
  tag.innerHTML = `<span class="rank">#${i + 1}</span>
    <div class="thumb"><img src="${e.thumbs.day3 ?? ''}" alt="Day 3 frame"><img class="d5" src="${e.thumbs.day5 ?? ''}" alt="Day 5 frame"><span class="lbl">day 3</span></div>
    <div class="score"><b>${Math.round(e.score * 100)}%</b><small>chance</small></div>
    <div class="outcome ${e.blastocyst ? 'yes' : 'no'}">${e.blastocyst ? 'Blastocyst' : 'Stalled'}</div>`
  $('#tags').append(tag)
  return { group: g, spec, inst, blast, tag, top: new THREE.Vector3(0, h + 1.3, 0), formed: !!e.blastocyst }
})

let revealed = false
$('#btn-reveal').addEventListener('click', () => {
  revealed = !revealed
  document.body.classList.toggle('revealed', revealed)
  document.querySelectorAll('.tag .lbl').forEach((s) => { s.textContent = revealed ? 'day 5' : 'day 3' })
  $('#btn-reveal span').textContent = revealed ? 'Back to day 3' : 'Show what happened by day 5'
  specimens.forEach((s) => {
    s.blast.visible = revealed && s.formed
    s.inst.visible = !(revealed && s.formed)
  })
})

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.setViewOffset(w, h, w > 1000 ? -w * 0.135 : 0, -h * 0.02, w, h)
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

const v = new THREE.Vector3()
const clock = new THREE.Clock()
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime()
  const w = window.innerWidth
  const h = window.innerHeight
  specimens.forEach((s, i) => {
    s.spec.rotation.y = t * 0.3 + i
    s.spec.position.y = s.top.y - 0.68 + 0.04 * Math.sin(t * 0.8 + i)
    v.copy(s.top).applyMatrix4(s.group.matrixWorld).project(camera)
    s.tag.style.transform = `translate(${(v.x + 1) / 2 * w}px, ${(1 - v.y) / 2 * h}px) translate(-50%, -100%)`
  })
  controls.update()
  renderer.render(scene, camera)
})
