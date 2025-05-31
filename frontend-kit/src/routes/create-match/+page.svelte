<script>
    import {onMount} from 'svelte';
    import {page} from '$app/stores';
    import {apiFetch} from '$lib/api';
    import { preventDefault } from 'svelte/legacy';
    import tpi from '../../assets/Barplot of Team Power Index.png'
    import chart from '../../assets/Bar chart of Match Result Distribution.png'

    let allTeams = [];
    let homeTeamId = null;
    let awayTeamId = null;

    let isLoadingTeams = true;
    let error = null;
    let successMessage = null;
    let isSubmitting = false;

    onMount(async () => {
        isLoadingTeams = true;
        try {
            allTeams = await apiFetch('/teams');
        } catch(err) {
            error = `Błąd ładowania drużyn: ${err.message}`;
            console.error(error);
        } finally {
            isLoadingTeams = false;
        }
    });

    async function handleAddFixture() {
        if (!homeTeamId || !awayTeamId) {
            error = "Dodanie drużyny gospodarzy oraz gości jest wymagane!"
            return;
        }
        if (homeTeamId === awayTeamId) {
            error = "Drużyna gospodarzy i gości nie może być taka sama.";
            return;
        }
        isSubmitting = true; error = null; successMessage = null;
        try {
            const newFixtureData = {
                home_team_id: parseInt(homeTeamId),
                away_team_id: parseInt(awayTeamId)
            }

            const createdMatch = await apiFetch('/predicted_matches/create', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(newFixtureData)
            });

            successMessage = 'Pomyślnie dodano mecz do predykcji!'
            homeTeamId = null;
            awayTeamId = null;

        } catch(err) {
            console.error("Error adding fixture: ", err);
            error = `Nie udało się dodać meczu: ${err.message}`;
        } finally {
            isSubmitting = false;
        }
    }

</script>

<svelte:head>
    <title>Dodaj Nowy mecz - AI Predictor</title>
</svelte:head>

<section class="add-fixture-page content-card">
    <h1>Dodaj Nowy Mecz do Systemu</h1>
    {#if isLoadingTeams}
        <p>Ładowanie danych formularza...</p>
    {:else if error && allTeams.length === 0}
        <p class="error-message">{error}</p>
    {:else}
        <form on:submit|preventDefault={handleAddFixture} class="fixture-form">
            <div class="form-group">
                <label for="home-team">Gospodarz:</label>
                <select id="home-team" bind:value={homeTeamId} required>
                    <option value={null}>-- Wybierz drużynę --</option>
                    {#each allTeams as team (team.id)}
                        <option value={team.id} disabled={team.id === awayTeamId}>{team.team_name}</option>
                    {/each}
                </select>
            </div>

            <div class="form-group">
                <label for="away-team">Gość:</label>
                <select id="away-team" bind:value={awayTeamId} required>
                    <option value={null}>-- Wybierz drużynę --</option>
                    {#each allTeams as team (team.id)}
                        <option value={team.id} disabled={team.id === homeTeamId}>{team.team_name}</option>
                    {/each}
                </select>
            </div>
            <button type="submit" disabled={isSubmitting}>
                {#if isSubmitting} Zapisywanie...{:else}Dodaj Mecz{/if}
            </button>
        </form>
        {#if successMessage}
            <p class="success-message" style="margin-top: 1rem;">{successMessage}</p>
        {/if}
        {#if error && !isSubmitting}
            <p class="error-message" style="margin-top: 1rem;">{error}</p>
        {/if}
    {/if}
    <div class="visualizations-area card" style="margin-top: 2rem;">
        <h2>Wizualizacje Ogólne</h2>
        <div class="viz-grid">
            <div class="viz-placeholder"><img src="{tpi}" alt="Wykres TPI"></div>
            <div class="viz-placeholder"><img src="{chart}" alt="Wykres"></div>
        </div>
    </div>
</section>

<style>
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .add-fixture-page h1 {
        color: #333;
        border-bottom: 2px solid #1abc9c;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .fixture-form {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        max-width: 600px;
        margin: 0 auto;
    }
    .form-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .form-group label {
        font-weight: 500;
        color: #555;
    }
    select, input[type="date"] {
        padding: 0.7rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 1rem;
    }
    .fixture-form button {
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        align-self: center;
    }
    .success-message {
        color: #27ae60;
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        padding: 0.75rem 1.25rem;
        border-radius: .25rem;
    }
    .visualizations-area {
        margin-top: 2rem;
        padding: 1.5rem;
    }
    .visualizations-area h2 {
        margin-top: 0;
        color: #16a085;
    }
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .viz-placeholder {
        border: 2px dashed #ccc;
        padding: 2rem;
        text-align: center;
        color: #777;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
    }
    img {
    max-width: 100%;
    max-height: 100%;
    }
    .error-message { color: red; }
</style>