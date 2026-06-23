#include "ardl_memory.h"
#include <stdio.h>  
#include <stdlib.h> // malloc and free functions

#ifndef __cplusplus
#define nullptr ((void*)0)
#endif


/*
#----------------------------
# Memory Area Functions
#---------------------------
*/
MemoryArena* arena_create(size_t size){
    MemoryArena *arena = malloc(sizeof(MemoryArena));
    arena->buffer = malloc(size);
    arena->size = size;
    arena->offset = 0;
    return arena;
}

void* arena_alloc(MemoryArena *arena, size_t size){
    size = (size + 7) & ~7; // 8-byte alignment for stable memory mapping

    if(arena->offset + size > arena->size){
        printf("Error (Memory): Arena out of memory!\n");
        return nullptr;
    }

    void *ptr = (void*)(arena->buffer + arena->offset);
    arena->offset += size;
    return ptr;
}

void arena_reset(MemoryArena *arena){
    arena->offset = 0;
}

void arena_free(MemoryArena *arena){
    free(arena->buffer);
    free(arena);
}
