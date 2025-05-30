<script>
    // Importy potrzebne dla layoutu, np. jeśli masz logikę w Navbar
    // import Navbar from '$lib/components/Navbar.svelte'; // Jeśli wydzielisz Navbar
</script>

<div class="app-container">
    <header>
        <nav class="navbar">
            <a href="/" class="nav-logo">⚽ Strona Główna</a>
            <ul class="nav-links">
                <li><a href="/matches" class="nav-item">Mecze</a></li>
                <li><a href="/season_stats/" class="nav-item">Statystyki drużyn</a></li>
                <li><a href="/predict" class="nav-item">Przewiduj Mecz</a></li>
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
    /* Twoje style globalne z App.svelte - te z tagiem <style global> */
    body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
            Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        line-height: 1.6;
        background-color: #f0f2f5; /* Dodane dla spójności z Twoim Dashboard */
    }
    /* Usunięto #app i .app-container z body, bo .app-container jest w tym pliku */
    /* Jeśli chcesz, aby .app-container był głównym kontenerem dla #app z src/app.html,
       możesz przenieść jego style do globalnego app.css lub zostawić tutaj,
       ale wtedy <div id="svelte"> w src/app.html (lub %sveltekit.body%) będzie jego rodzicem. */

    /* Style dla paska nawigacyjnego i stopki - mogą być tutaj lub w osobnym pliku CSS
       importowanym globalnie, lub w dedykowanych komponentach Navbar/Footer */
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
        padding: 1rem 0; /* Zamiast height, dla elastyczności */
    }
    .nav-logo {
        font-size: 1.6rem;
        font-weight: bold;
        color: white; /* Zmieniono na biały dla lepszego kontrastu z tłem headera */
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
        color: #ecf0f1; /* Jaśniejszy dla linków */
        text-decoration: none;
        padding: 0.6rem 0.8rem;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    /* Aktywny link w SvelteKit często obsługuje się inaczej, np. przez store $page.url.pathname
       lub dedykowane atrybuty, ale proste :hover może wystarczyć na początek.
       Dla `aria-current="page"` SvelteKit dodaje tę klasę automatycznie. */
    .nav-item:hover,
    a[aria-current="page"] { /* SvelteKit dodaje aria-current="page" do aktywnego linku */
        background-color: #34495e;
        color: white;
    }
    main {
        flex-grow: 1;
        padding: 1.5rem 2rem;
        max-width: 1500px;
        width: 100%;
        margin: 1rem auto; /* Odstęp od góry */
        box-sizing: border-box;
    }
    footer {
        text-align: center;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: auto; /* Jeśli chcesz, aby stopka była na dole */
    }

</style>