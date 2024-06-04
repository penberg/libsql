/*
** 2024-06-04
**
** Copyright 2024 the libSQL authors
**
** Permission is hereby granted, free of charge, to any person obtaining a copy of
** this software and associated documentation files (the "Software"), to deal in
** the Software without restriction, including without limitation the rights to
** use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
** the Software, and to permit persons to whom the Software is furnished to do so,
** subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
** FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
** COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
** IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**
******************************************************************************
**
** 1-bit vector format utilities.
*/
#ifndef SQLITE_OMIT_VECTOR
#include "sqliteInt.h"

#include "vectorInt.h"

#include <math.h>

/**************************************************************************
** Utility routines for debugging
**************************************************************************/

void vector1BitDump(Vector *pVec){
  u8 *elems = pVec->data;
  unsigned i, j;
  for(i = 0; i < pVec->dims/8; i++){
    for(j = 0; j < 8; j++){
      printf("%d ", (elems[i] >> j) & 1);
    }
  }
  printf("\n");
}

/**************************************************************************
** Utility routines for vector serialization and deserialization
**************************************************************************/

size_t vector1BitSerializeToBlob(
  Vector *v,
  unsigned char *blob,
  size_t blobSz
){
  unsigned char *blobPtr = blob;
  u8 *elems = v->data;
  assert( blobSz >= vectorDataSize(v->type, v->dims) );
  memcpy(blob, elems, vectorDataSize(v->type, v->dims));
  return vectorDataSize(v->type, v->dims);
}

size_t vector1BitDeserializeFromBlob(
  Vector *v,
  const unsigned char *blob,
  size_t blobSz
){
  u8 *elems = v->data;
  assert( blobSz >= vectorDataSize(v->type, v->dims) );
  memcpy(elems, blob, vectorDataSize(v->type, v->dims));
  return vectorDataSize(v->type, v->dims);
}

void vector1BitSerialize(
  sqlite3_context *context,
  Vector *v
){
  float *elems = v->data;
  unsigned char *blob;
  unsigned int blobSz;

  blobSz = vectorDataSize(v->type, v->dims);
  blob = contextMalloc(context, blobSz);

  if( blob ){
    vector1BitSerializeToBlob(v, blob, blobSz);

    sqlite3_result_blob(context, (char*)blob, blobSz, sqlite3_free);
  } else {
    sqlite3_result_error_nomem(context);
  }
}

void vector1BitDeserialize(
  sqlite3_context *context,
  Vector *v
){
  float *elems = v->data;
  unsigned bufSz;
  unsigned bufIdx = 0;
  char *z;

  bufSz = 2 + v->dims * 33;
  z = contextMalloc(context, bufSz);

  if( z ){
    unsigned i;

    z[bufIdx++]= '[';
    for (i = 0; i < v->dims; i++) { 
      char tmp[12];
      unsigned bytes = formatF32(elems[i], tmp);
      memcpy(&z[bufIdx], tmp, bytes);
      bufIdx += strlen(tmp);
      z[bufIdx++] = ',';
    }
    bufIdx--;
    z[bufIdx++] = ']';

    sqlite3_result_text(context, z, bufIdx, sqlite3_free);
  } else {
    sqlite3_result_error_nomem(context);
  }
}

Vector* vectorConvertTo1Bit(Vector *v){
  float *src;
  Vector *p;
  int i, j;
  u8 *dst;

  assert( v->type == VECTOR_TYPE_FLOAT32 );
  p = vectorAlloc(VECTOR_TYPE_1BIT, v->dims);
  if( !p ) return 0;
  src = v->data;
  dst = p->data;
  memset(dst, 0, v->dims/8);
  for(i = 0; i < v->dims/8; i++){
    for(j = 0; j < 8; j++){
      int v1 = src[i + j] > 0;
      dst[i] |= ((v1 >> j) & 1) << j;
    }
  }
  return p;
}

float vector1BitDistanceHamming(Vector *v1, Vector *v2){
  u8 *p1 = v1->data;
  u8 *p2 = v2->data;
  unsigned i;
  unsigned distance = 0;
  for(i = 0; i < v1->dims/8; i++){
    distance += __builtin_popcount(p1[i] ^ p2[i]);
  }
  // TODO: do we want to normalize this?
  return distance / (float)v1->dims;
}

void vector1BitInitFromBlob(Vector *p, const unsigned char *blob, size_t blobSz){
  p->dims = blobSz / 8;
  p->data = (void*)blob;
}

int vector1BitParseBlob(
  sqlite3_value *arg,
  Vector *v,
  char **pzErr
){
  const unsigned char *blob;
  float *elems = v->data;
  char zErr[128];
  unsigned i;
  size_t len;

  if( sqlite3_value_type(arg)!=SQLITE_BLOB ){
    *pzErr = sqlite3_mprintf("invalid vector: not a blob type");
    goto error;
  }

  blob = sqlite3_value_blob(arg);
  if( !blob ) {
    *pzErr = sqlite3_mprintf("invalid vector: zero length");
    goto error;
  }
  len = sqlite3_value_bytes(arg) / sizeof(float);
  if (len > MAX_VECTOR_SZ) {
    *pzErr = sqlite3_mprintf("invalid vector: too large: %d", len);
    goto error;
  }
  for(i = 0; i < len; i++){
    if( !blob ){
      *pzErr = sqlite3_mprintf("malformed blob");
      goto error;
    }
    elems[i] = *blob;
    blob += sizeof(u8);
  }
  v->dims = len;
  return len;
error:
  return -1;
}

#endif /* !defined(SQLITE_OMIT_VECTOR) */
