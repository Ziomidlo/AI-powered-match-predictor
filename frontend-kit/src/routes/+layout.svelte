<script>
</script>

<div class="app-container">
    <header>
        <nav class="navbar">
            <a href="/" class="nav-logo">Strona Główna</a>
            <ul class="nav-links">
                <li><a href="/matches" class="nav-item">Mecze</a></li>
                <li><a href="/season-stats/" class="nav-item">Statystyki drużyn</a></li>
                <li><a href="/create-match" class="nav-item">Stwórz mecz</a></li>
                <li><a href="/predictions" class="nav-item">Historia Predykcji</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <slot />
    </main>

    <footer>
        <p>&copy; {new Date().getFullYear()} AI Football Predictor.</p>
    </footer>
</div>

<style global>
    body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
            Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        line-height: 1.6;
        background-color: #f0f2f5;
    }
    .app-container {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    header {
        background-color: #2c3e50;
        color: white;
        padding: 0 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem 0;
    }
    .nav-logo {
        font-size: 1.6rem;
        font-weight: bold;
        color: white;
        text-decoration: none;
    }
    .nav-links {
        list-style: none;
        display: flex;
        gap: 1.2rem;
        padding: 0;
        margin: 0;
    }
    .nav-item {
        color: #ecf0f1;
        text-decoration: none;
        padding: 0.6rem 0.8rem;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }

    .nav-item:hover,
    a[aria-current="page"] {
        background-color: #34495e;
        color: white;
    }
    main {
        flex-grow: 1;
        padding: 1.5rem 2rem;
        max-width: 1500px;
        width: 100%;
        margin: 1rem auto;
        box-sizing: border-box;
    }
    footer {
        text-align: center;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: auto;
    }

</style>