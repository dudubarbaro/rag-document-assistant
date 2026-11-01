import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Em desenvolvimento, /api é redirecionado para o backend FastAPI
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
    // permite abrir o dev server pelo endereço público do GitHub Codespaces
    allowedHosts: [".app.github.dev"],
  },
});
