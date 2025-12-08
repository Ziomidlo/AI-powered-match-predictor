<script>
    import { onMount } from 'svelte';
    import { apiFetch } from '../lib/api.js'; 
    import SeasonSelector from '../components/SeasonSelector.svelte';
    import LastMatches from '../components/LastMatches.svelte';
    import LeagueTable from '../components/LeagueTable.svelte';
    import ImageModal from '../components/ImageModal.svelte';

    import avgPosition from '../assets/Barplot of Average Position Overall by Team.png'
    import avgPoints from '../assets/Barplot of Average Points Per Game by Team.png'

    let isModalOpen = false;
    let modalImageSrc = '';
    let modalImageAlt = '';

    function openModal(src, alt) {
        modalImageSrc = src;
        modalImageAlt = alt;
        isModalOpen = true;
    }

    function closeModal() {
        isModalOpen = false;
    }

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

{#if isModalOpen}
    <ImageModal 
        imageSrc={modalImageSrc} 
        imageAlt={modalImageAlt}
        on:close={closeModal} 
    />
{/if}


{#if seasons.length > 0}
    <div class="dashboard-grid">
        <div class="left card">
            <SeasonSelector 
                availableSeasons={seasons} 
                bind:selectedSeasonId={selectedSeasonId}
            />
            {#if selectedSeasonId !== null}
                <LeagueTable seasonId={selectedSeasonId} />
            {:else}
                <p>Wybierz sezon, aby zobaczyć tabelę ligową.</p>
            {/if}
        </div>
        <div class="right card">
            <LastMatches />
            
            <div class="visualizations-wrapper">
                <div class="chart-item">
                    <img src={avgPosition} alt="Wykres średniej pozycji">
                    <button class="zoom-btn" on:click={() => openModal(avgPosition, 'Wykres średniej pozycji')} title="Powiększ">
                        &#128269;
                    </button>
                </div>
                <div class="chart-item">
                    <img src={avgPoints} alt="Wykres średniej ilości punktów" >
                    <button class="zoom-btn" on:click={() => openModal(avgPoints, 'Wykres średniej ilości punktów')} title="Powiększ">
                        &#128269;
                    </button>
                </div>
            </div>

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
    .left {
        text-align: center;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }


    .right {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .visualizations-wrapper {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        margin-top: 1rem;
        border-top: 1px solid #eee;
        padding-top: 1.5rem;
    }

    .chart-item {
        position: relative;
        overflow: hidden;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        cursor: pointer;
    }

    .chart-item img {
        display: block;
        width: 100%;
        height: auto;
    }

    .zoom-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background-color 0.3s ease, opacity 0.3s ease;
        opacity: 0;
    }

    .chart-item:hover .zoom-btn {
        opacity: 1;
    }

    .zoom-btn:hover {
        background-color: rgba(0, 0, 0, 0.8);
    }
    

    img {
        max-width: 100%;
        max-height: 100%;
    }
</style>