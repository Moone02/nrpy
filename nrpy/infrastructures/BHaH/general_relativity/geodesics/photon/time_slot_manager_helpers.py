"""
Generates the C data structures and helper functions for the Time Slot Manager.

The Time Slot Manager implements a temporal "bucket sort" strategy. It divides
the simulation timeline into discrete slots, allowing the numerical pipeline
to process photons in parallel bundles based on their coordinate time.

Author: Dalton J. Moone
"""

from nrpy.infrastructures.BHaH import BHaH_defines_h as Bdefines_h


def time_slot_manager_helpers() -> None:
    """
    Generate and register the TimeSlotManager C code for injection into BHaH_defines.h.

    This function defines the PhotonList and TimeSlotManager structures and
    the static inline functions required to manage them.
    """
    c_code = r"""
// =============================================
// NRPy-Generated Time Slot Manager
// =============================================

/**
 * @brief A dynamic array holding indices of photons residing in a specific time slot.
 */
typedef struct {
    long int *photons;  ///< Array of photon indices.
    long int count;     ///< Current number of photons in this slot.
    long int capacity;  ///< Allocated capacity of the array.
} PhotonList;

/**
 * @brief The main manager for the Iterative Time Slotting algorithm.
 */
typedef struct {
    double t_min;         ///< Lower bound of the time domain.
    double t_max;         ///< Upper bound of the time domain.
    double delta_t_slot;  ///< Width of each time slot.
    int num_slots;        ///< Total count of slots.
    PhotonList *slots;    ///< Array of PhotonLists.
} TimeSlotManager;

/**
 * @brief Initializes the Time Slot Manager.
 */
static inline void slot_manager_init(TimeSlotManager *tsm, double t_min, double t_max, double delta_t_slot) {
    tsm->t_min = t_min;
    tsm->t_max = t_max;
    tsm->delta_t_slot = delta_t_slot;
    tsm->num_slots = (int)ceil((t_max - t_min) / delta_t_slot);

    if (tsm->num_slots <= 0) {
        fprintf(stderr, "Error: Invalid TimeSlotManager dimensions.\n");
        exit(1);
    }

    tsm->slots = (PhotonList *)malloc(sizeof(PhotonList) * tsm->num_slots);
    if (!tsm->slots) { exit(1); }

    for (int i = 0; i < tsm->num_slots; i++) {
        tsm->slots[i].capacity = 16;
        tsm->slots[i].count = 0;
        tsm->slots[i].photons = (long int *)malloc(sizeof(long int) * tsm->slots[i].capacity);
        if (!tsm->slots[i].photons) { exit(1); }
    }
}

static inline void slot_manager_free(TimeSlotManager *tsm) {
    if (!tsm || !tsm->slots) return;
    for (int i = 0; i < tsm->num_slots; i++) {
        free(tsm->slots[i].photons);
    }
    free(tsm->slots);
}

/**
 * @brief Maps coordinate time to a slot index.
 */
static inline int slot_get_index(const TimeSlotManager *tsm, double t) {
    if (t < tsm->t_min || t >= tsm->t_max) return -1;
    return (int)floor((t - tsm->t_min) / tsm->delta_t_slot);
}

/**
 * @brief Adds a photon index with geometric memory growth.
 */
static inline void slot_add_photon(PhotonList *slot, long int photon_idx) {
    if (slot->count >= slot->capacity) {
        long int new_capacity = slot->capacity * 2;
        long int *new_ptr = (long int *)realloc(slot->photons, sizeof(long int) * new_capacity);
        if (!new_ptr) {
            fprintf(stderr, "Error: Failed to realloc photon slot.\n");
            exit(1);
        }
        slot->photons = new_ptr;
        slot->capacity = new_capacity;
    }
    slot->photons[slot->count++] = photon_idx;
}

/**
 * @brief Removes a chunk of photons. Exits on count mismatch to prevent 
 * the integrator from processing garbage data.
 */
static inline void slot_remove_chunk(PhotonList *slot, long int *chunk_buffer, long int chunk_size) {
    if (chunk_size > slot->count) {
        fprintf(stderr, "Error: Slot underflow. Requested %ld, available %ld.\n", chunk_size, slot->count);
        exit(1);
    }

    for (long int i = 0; i < chunk_size; ++i) {
        chunk_buffer[i] = slot->photons[i];
    }

    long int remaining = slot->count - chunk_size;
    if (remaining > 0) {
        memmove(slot->photons, slot->photons + chunk_size, remaining * sizeof(long int));
    }
    slot->count -= chunk_size;
}
"""

    # Register the C code block to BHaH_defines.h
    Bdefines_h.register_BHaH_defines("time_slot_manager", c_code)
