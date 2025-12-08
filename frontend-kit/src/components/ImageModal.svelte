<script>
    import { createEventDispatcher, onMount, onDestroy } from 'svelte';

    export let imageSrc = '';
    export let imageAlt = 'Powiększona wizualizacja';

    const dispatch = createEventDispatcher();

    function closeModal() {
        dispatch('close');
    }

    function handleKeydown(event) {
        if (event.key === 'Escape') {
            closeModal();
        }
    }

</script>

<svelte:window on:keydown={handleKeydown}/>

<div class="modal-backdrop" on:click={closeModal}>
    <div class="modal-content" on:click|stopPropagation>
       <button class="close-button" on:click={closeModal} title="Zamknij (Esc)">
            &times;
        </button>
        <img src={imageSrc} alt={imageAlt} class="modal-image"/>
    </div>
</div>

<style>
    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.75);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        cursor: pointer;
    }

    .modal-content {
        position: relative;
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        cursor: default;
        max-width: 90vw;
        max-height: 90vh;
    }

    .modal-image {
        display: block;
        max-width: 100%;
        max-height: calc(90vh - 20px);
        border-radius: 4px;
    }

    .close-button {
        position: absolute;
        top: -15px;
        right: -15px; 
        width: 35px;
        height: 35px;
        border-radius: 50%;
        border: none;
        background-color: #333;
        color: white;
        font-size: 24px;
        line-height: 35px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.2s ease;
    }

    .close-button:hover {
        transform: scale(1.1);
    }
</style>