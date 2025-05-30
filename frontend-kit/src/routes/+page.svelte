<script>
    import { onMount } from 'svelte';
    // W SvelteKit, jeśli api.js jest w src/lib, importujesz go przez $lib
    import { apiFetch } from '../lib/api.js'; 
    // Importuj swoje komponenty (zakładam, że są w src/lib/components/)
    import SeasonSelector from '../components/SeasonSelector.svelte';
    import LastMatches from '../components/LastMatches.svelte';
    import LeagueTable from '../components/LeagueTable.svelte';
    import avgPosition from '../assets/Barplot of Average Position Overall by Team.png'
    import avgPoints from '../assets/Barplot of Average Points Per Game by Team.png'


    // Ścieżka do obrazka - jeśli jest w static/, to ścieżka jest od root
    // Jeśli jest w src/assets i importujesz go, Vite/SvelteKit zajmie się ścieżką
    // Dla przykładu załóżmy, że MatchResults jest w static/visualizations/

    let selectedSeasonId = 1; // Zmieniono nazwę, żeby było jasne, że to ID
    let seasons = []; // Powinno zawierać obiekty { id: 1, season_name: "2023/2024" }

    onMount(async () => {
    try {
        const data = await apiFetch("/seasons"); 
        if (data && data.length > 0) {
            // (2) Przekształć dane sezonów na format {id, name}
            // Zakładając, że API zwraca np. [{id: 1, season: "2023/2024"}, ...]
            seasons = data.map(s => ({ id: s.id, name: s.season })); 
            // (3) Ustaw domyślny selectedSeasonId (np. ostatni z listy)
            if (seasons.length > 0) {
                 // Zakładając, że API zwraca sezony posortowane (np. najnowszy na końcu lub początku)
                 // Jeśli sortowanie jest rosnące po ID lub nazwie:
                 selectedSeasonId = 1
                 // Jeśli sortowanie jest malejące:
                 // selectedSeasonId = seasons[0].id;
            }
        }
    } catch (e) {
        console.error("Error fetching seasons for dashboard:", e);
    }

    function handleSeasonChange(event) {
    // event.detail teraz powinno być ID sezonu (lub null) z poprawionego SeasonSelector
    selectedSeasonId = event.detail; 
    console.log("Dashboard: Zmieniono sezon na ID:", selectedSeasonId); // Do debugowania
}
});


</script>

<svelte:head>
    <title>Dashboard - AI Football Predictor</title>
</svelte:head>

{#if seasons.length > 0}
    <div class="dashboard-grid">
        <div class="left card">
        <SeasonSelector 
            availableSeasons={seasons} 
            bind:selectedSeasonId={selectedSeasonId}
        />
        {#if selectedSeasonId !== null} <LeagueTable seasonId={selectedSeasonId} />
            {:else}
                <p>Wybierz sezon, aby zobaczyć tabelę ligową.</p>
        {/if}
        </div>
        <div class="right card">
            <LastMatches />
            <img src={avgPosition} alt="Wykres średniej pozycji">
            <img src={avgPoints} alt="Wykres średniej ilości punktów" >
        </div>
    </div>
{:else}
    <p>Ładowanie danych sezonów lub brak sezonów do wyświetlenia...</p>
{/if}



<style>
    /* Style dla Dashboard.svelte (te które były wcześniej) */
    .dashboard-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr; /* Dostosuj proporcje */
        gap: 2rem;
    }
    .left{
        text-align: center;
        /* Usunięto powtarzające się style, zakładając, że .card jest zdefiniowane globalnie lub w layoucie */
    }
    .card { /* Dodaj ten styl do globalnego app.css lub +layout.svelte <style global> */
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .chart-container {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .chart-container h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    img {
    max-width: 100%;
    max-height: 100%;
    }
</style>



