import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  build: {
    outDir: "../static/app",
    emptyOutDir: true,
    minify: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/arxie-react-app.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/arxie-react-app[extname]"
      }
    }
  }
});
