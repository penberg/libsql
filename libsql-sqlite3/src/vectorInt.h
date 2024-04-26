#ifndef _VECTOR_H
#define _VECTOR_H

#include "sqlite3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Objects */
typedef struct Vector Vector;
typedef struct DiskAnnIndex DiskAnnIndex;

#define MAX_VECTOR_SZ 16000

#define VECTOR_TYPE_F32 0

#define VECTOR_FLAGS_STATIC 1

/* An instance of this object represents a vector.
*/
struct Vector {
  u16 type;
  u16 flags;
  u32 len;
  void *data;
};

size_t vectorElemSize(u16);
Vector *vectorAlloc(u16, u32);
void vectorFree(Vector *v);
int vectorParse(sqlite3_value *, Vector *, char **);
size_t vectorSerializeToBlob(Vector *, unsigned char *, size_t);
size_t vectorDeserializeFromBlob(Vector *, const unsigned char *, size_t);
void vectorDump(Vector *v);
float vectorDistanceCos(Vector *, Vector *);

void vectorF32Dump(Vector *v);
void vectorF32Deserialize(sqlite3_context *,Vector *v);
void vectorF32Serialize(sqlite3_context *,Vector *v);
void vectorF32InitFromBlob(Vector *, const unsigned char *);
int vectorF3ParseBlob(sqlite3_value *, Vector *, char **);
size_t vectorF32SerializeToBlob(Vector *, unsigned char *, size_t);
size_t vectorF32DeserializeFromBlob(Vector *, const unsigned char *, size_t);
float vectorF32DistanceCos(Vector *, Vector *);

int diskAnnCreateIndex(sqlite3 *, const char *zName, unsigned int);
int diskAnnOpenIndex(sqlite3 *, const char *zName, DiskAnnIndex **);
void diskAnnCloseIndex(DiskAnnIndex *pIndex);
int diskAnnInsert(DiskAnnIndex *, Vector *v, i64);
int diskAnnSearch(DiskAnnIndex *, Vector*, unsigned int, i64*);

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* _VECTOR_H */