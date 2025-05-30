<script>
  export let availableSeasons = [];
  export let selectedSeasonId = null;
  export let onSeasonChange = () => {};

  function handleChange(e) {
    const selectedId = parseInt(e.target.value);
    onSeasonChange(selectedId);
  }
</script>


<select bind:value={selectedSeasonId} on:change={handleChange}>
  <option value={null}>Wybierz sezon...</option>
  {#each availableSeasons as season (season.id)}
    <option value={season.id}>{season.name}</option>
  {/each}
</select>
