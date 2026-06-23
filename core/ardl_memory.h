#ifndef ARDL_MEMORY_H
#define ARDL_MEMORY_H

#include <stddef.h> 

// --- Memory Structures ---
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
} MemoryArena;


// --- 1. Memory Management ---
MemoryArena* arena_create(size_t size);
void* arena_alloc(MemoryArena *arena, size_t size);
void arena_reset(MemoryArena *arena);
void arena_free(MemoryArena *arena);


#endif
