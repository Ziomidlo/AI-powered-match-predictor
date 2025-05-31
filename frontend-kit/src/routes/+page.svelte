<script>
    import { onMount } from 'svelte';
    import { apiFetch } from '../lib/api.js'; 
    import SeasonSelector from '../components/SeasonSelector.svelte';
    import LastMatches from '../components/LastMatches.svelte';
    import LeagueTable from '../components/LeagueTable.svelte';
    import avgPosition from '../assets/Barplot of Average Position Overall by Team.png'
    import avgPoints from '../assets/Barplot of Average Points Per Game by Team.png'


    let selectedSeasonId = null;
    let seasons = []; 
    onMount(async () => {
    try {
        const data = await apiFetch("/seasons"); 
        if (data && data.length > 0) {
            seasons = data.map(s => ({ id: s.id, name: s.season })); 
            if (seasons.length > 0) {
                 selectedSeasonId = seasons[0].id;
            }
        }
    } catch (e) {
        console.error("Error fetching seasons for dashboard:", e);
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
    .dashboard-grid {
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 2rem;
    }
    .left{
        text-align: center;
    }
    .card {
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



