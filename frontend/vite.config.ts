import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  envDir: "..",
  build: {
    rollupOptions: {
      input: { main: "index.html", phase10: "phase10.html" },
    },
  },
});
