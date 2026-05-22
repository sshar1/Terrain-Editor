// ============================================================
//  Input handling via Hammer.js
//  Exposes window.inputState for engine.js to consume
// ============================================================

window.inputState = {
    // Per-frame rotation deltas (radians), consumed & reset by engine each frame
    deltaX: 0,
    deltaY: 0,

    // Velocity for momentum after release
    velocityX: 0,
    velocityY: 0,

    // Camera distance (for pinch zoom)
    zoom: 7,

    // Whether user is currently interacting
    interacting: false,
};

(function () {
    const canvas = document.getElementById("canvas");
    if (!canvas) {
        console.error("Canvas element not found!");
        return;
    }
    const mc = new Hammer.Manager(canvas);

    // ---- Recognizers ----
    mc.add(new Hammer.Pan({ direction: Hammer.DIRECTION_ALL, threshold: 1 }));
    mc.add(new Hammer.Pinch({ enable: true }));
    mc.get('pinch').recognizeWith('pan');

    // ---- Pan → Rotation deltas ----
    let prevDeltaX = 0;
    let prevDeltaY = 0;

    mc.on('panstart', function () {
        prevDeltaX = 0;
        prevDeltaY = 0;
        window.inputState.interacting = true;
    });

    mc.on('panmove', function (e) {
        const dx = e.deltaX - prevDeltaX;
        const dy = e.deltaY - prevDeltaY;
        prevDeltaX = e.deltaX;
        prevDeltaY = e.deltaY;

        const sensitivity = 0.007;
        // Accumulate deltas — engine will consume and reset them
        window.inputState.deltaX += dx * sensitivity;
        window.inputState.deltaY += dy * sensitivity;

        // Store velocity for momentum
        window.inputState.velocityX = dx * sensitivity;
        window.inputState.velocityY = dy * sensitivity;
    });

    mc.on('panend pancancel', function () {
        window.inputState.interacting = false;
    });

    // ---- Pinch → Zoom ----
    let startZoom = 7;

    mc.on('pinchstart', function () {
        startZoom = window.inputState.zoom;
        window.inputState.interacting = true;
    });

    mc.on('pinchmove', function (e) {
        const amplified = Math.pow(e.scale, 20);
        window.inputState.zoom = Math.min(20, Math.max(2, startZoom / amplified));
    });

    mc.on('pinchend pinchcancel', function () {
        window.inputState.interacting = false;
    });

    // ---- Mouse wheel fallback for desktop zoom ----
    canvas.addEventListener('wheel', function (e) {
        e.preventDefault();
        window.inputState.zoom = Math.min(20, Math.max(2, window.inputState.zoom + e.deltaY * 0.01));
    }, { passive: false });
})();