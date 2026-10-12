// The pages are laid out for laptop and desktop screens, so phones get a notice
// first (styled by .small-screen in base.css), once per visit.
const KEY = 'embryoscan-small-screen-ok'
let dismissed = false
try {
  dismissed = sessionStorage.getItem(KEY) === '1'
} catch {
  // storage blocked: show the notice
}
if (!dismissed) {
  const notice = document.createElement('div')
  notice.className = 'small-screen'
  notice.innerHTML = `<div class="card">
    <i class="ti ti-device-desktop"></i>
    <h2>Best on a bigger screen</h2>
    <p>Embryo Scan is an interactive 3D lab: real embryo time-lapse videos, an AI grader and 3D embryos you can turn around. It's laid out for a laptop or desktop.</p>
    <button type="button" class="primary">Continue anyway</button>
  </div>`
  notice.querySelector('button').addEventListener('click', () => {
    notice.hidden = true
    try {
      sessionStorage.setItem(KEY, '1')
    } catch {
      // not remembered; the notice returns on the next page
    }
  })
  document.body.append(notice)
}
