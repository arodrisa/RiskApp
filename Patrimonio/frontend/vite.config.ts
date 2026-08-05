import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  server: {
    proxy: {
      '/owners': 'http://127.0.0.1:8001',
      '/assets': 'http://127.0.0.1:8001',
      '/positions': 'http://127.0.0.1:8001',
      '/dashboard': 'http://127.0.0.1:8001',
      '/prices': 'http://127.0.0.1:8001',
      '/auth': 'http://127.0.0.1:8001',
      '/audit-log': 'http://127.0.0.1:8001',
      '/investing-assets': 'http://127.0.0.1:8001',
      '/entity-ownerships': 'http://127.0.0.1:8001',
      '/project-users': 'http://127.0.0.1:8001',
      '/project-invitations': 'http://127.0.0.1:8001',
      '/restore': 'http://127.0.0.1:8001',
      '/export': 'http://127.0.0.1:8001',
      '/health': 'http://127.0.0.1:8001'
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
