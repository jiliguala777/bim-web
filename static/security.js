/* Attach CSRF tokens only to same-origin requests that change state. */
(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, options = {}) => {
        const request = new Request(input, options);
        const url = new URL(request.url, window.location.href);
        if (url.origin === window.location.origin && !/^(GET|HEAD|OPTIONS|TRACE)$/i.test(request.method)) {
            const token = document.querySelector('meta[name="csrf-token"]')?.content;
            const headers = new Headers(request.headers);
            if (token) headers.set('X-CSRFToken', token);
            return originalFetch(new Request(request, { headers }));
        }
        return originalFetch(request);
    };
})();
