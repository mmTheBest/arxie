// Legacy compatibility shim: keep `/ui/paperbase-ui.js` available for callers
// that predate the build-free ES module entrypoint at `/ui/js/app.js`.
const arxieModuleScript = document.createElement("script");
arxieModuleScript.type = "module";
arxieModuleScript.src = "/ui/js/app.js";
document.head.appendChild(arxieModuleScript);
