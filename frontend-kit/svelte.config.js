import adapter from '@sveltejs/adapter-auto';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter(),
		alias: {
			$components: './src/components'
		}
	},
	vite: {
      server: {
        proxy: {
          '/api': 'http://localhost:8000'
        }
      }
    }
};

export default config;
