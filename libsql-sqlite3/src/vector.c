/*
** 2024-03-18
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
** libSQL vector search.
*/
#ifndef SQLITE_OMIT_VECTOR
#include "sqliteInt.h"

#include "vectorInt.h"

#define MAX_VECTOR_SZ 16000
#define MAX_FLOAT_CHAR_SZ  1024

/**************************************************************************
** Utility routines for dealing with Vector objects
**************************************************************************/

size_t vectorElemSize(u16 type){
  switch (type) {
    case VECTOR_TYPE_F32:
      return sizeof(float);
    default:
      assert(0);
  }
  return 0;
}


/*
** Initialize the Vector object
*/
static void vectorInit(Vector *p, u32 type, void *data, unsigned int len){
  p->data = data;
  p->type = type;
  p->len = len;
}

/**
** Allocate a Vector object and its data buffer.
**/
Vector *vectorAlloc(u16 type, u32 len){
  void *p;

  p = sqlite3_malloc(sizeof(Vector) + vectorElemSize(type) * len);
  if( p==NULL ){
    return NULL;
  }
  vectorInit(p, type, p + sizeof(Vector), len);
  return p;
}

/**
** Allocate a Vector object and its data buffer from the SQLite context. 
**/
static Vector *vectorContextAlloc(sqlite3_context *pCtx, u32 type){
  void *p;

  p = contextMalloc(pCtx, sizeof(Vector) + MAX_VECTOR_SZ);
  if( p==NULL ){
    sqlite3_result_error_nomem(pCtx);
    return NULL;
  }
  vectorInit(p, type, p + sizeof(Vector), MAX_VECTOR_SZ);
  return p;
}

/**
** Free a Vector object and its data buffer allocated, unless the vector is static.
**/
void vectorFree(Vector *p){
  if( p->flags & VECTOR_FLAGS_STATIC ){
    return;
  }
  sqlite3_free(p);
}

/*
** Initialize a static Vector object.
**
** Note that that the vector object points to the blob so if
** you free the blob, the vector becomes invalid.
**/
static void vectorInitStatic(Vector *p, u32 type, const unsigned char *blob){
  switch (type) {
    case VECTOR_TYPE_F32:
      vectorF32InitFromBlob(p, blob);
      break;
    default:
      assert(0);
  }
  p->type = type;
  p->flags = VECTOR_FLAGS_STATIC;
}

float vectorDistanceCos(Vector *v1, Vector *v2){
  assert(v1->type == v2->type);
  switch (v1->type) {
    case VECTOR_TYPE_F32:
      return vectorF32DistanceCos(v1, v2);
      break;
    default:
      assert(0);
  }
  return -1;
}

static int vectorParseText(
  sqlite3_value *arg,
  Vector *v,
  char **pzErrMsg
){
  char elBuf[MAX_FLOAT_CHAR_SZ];
  const unsigned char *zStr;
  float *elems = v->data;
  char zErr[128];
  int bufidx = 0;
  int vecidx = 0;
  double el;

  if( sqlite3_value_type(arg)!=SQLITE_TEXT ){
    *pzErrMsg = sqlite3_mprintf("invalid vector: not a text type");
    goto error;
  }

  memset(elBuf, 0, sizeof(elBuf));

  zStr = sqlite3_value_text(arg);

  while (zStr && sqlite3Isspace(*zStr))
    zStr++;

  if( zStr==0 ) return 0;

  if (*zStr != '[') {
    sqlite3_snprintf(sizeof(zErr), zErr, "invalid vector: doesn't start with ']':");
    goto error;
  }
  zStr++;

  while (zStr != NULL && *zStr != '\0' && *zStr != ']') {
    char this = *zStr++;
    if (sqlite3Isspace(this)) {
      continue;
    }
    if (this != ',' && this != ']') {
      elBuf[bufidx++] = this;
      if (bufidx > MAX_FLOAT_CHAR_SZ) {
        char zErr[MAX_FLOAT_CHAR_SZ+100];
        sqlite3_snprintf(sizeof(zErr), zErr, "float too big while parsing vector: %s...", elBuf);
        return -1;
      }
    } else {
      if (sqlite3AtoF(elBuf, &el, bufidx, SQLITE_UTF8) <= 0) {
        sqlite3_snprintf(sizeof(zErr), zErr, "invalid number: %s...", elBuf);
        return -1;
      }
      bufidx = 0;
      memset(elBuf, 0, sizeof(elBuf));
      elems[vecidx++] = el;
      if (vecidx >= MAX_VECTOR_SZ) {
        sqlite3_snprintf(sizeof(zErr), zErr, "vector is larger than the maximum: (%d)", MAX_VECTOR_SZ);
        return -1;
      }
    }
  }
  if (bufidx != 0) {
    if (sqlite3AtoF(elBuf, &el, bufidx, SQLITE_UTF8) <= 0) {
      sqlite3_snprintf(sizeof(zErr), zErr, "invalid number: %s...", elBuf);
        return -1;
    }
    elems[vecidx++] = el;
    if (vecidx >= MAX_VECTOR_SZ) {
      sqlite3_snprintf(sizeof(zErr), zErr, "vector is larger than the maximum: (%d)", MAX_VECTOR_SZ);
        return -1;
    }
  }
  if (zStr && *zStr!= ']') {
    sqlite3_snprintf(sizeof(zErr), zErr, "malformed vector, doesn't end with ']'");
    return -1;
  }
  v->len = vecidx;
  return vecidx;
error:
  return -1;
}

static int vectorParseBlob(
  sqlite3_value *arg,
  Vector *v,
  char **pzErrMsg
){
  switch (v->type) {
    case VECTOR_TYPE_F32:
      return vectorF3ParseBlob(arg, v, pzErrMsg);
      break;
    default:
      assert(0);
  }
  return -1;
}

int vectorParse(
  sqlite3_value *arg,
  Vector *v,
  char **pzErrMsg
){
  switch( sqlite3_value_type(arg) ){
    case SQLITE_BLOB:
      return vectorParseBlob(arg, v, pzErrMsg);
      break;
    case SQLITE_TEXT:
      return vectorParseText(arg, v, pzErrMsg);
    default:
      *pzErrMsg = sqlite3_mprintf("invalid vector: not a text or blob type");
      return -1;
  }
}

static inline int isInteger(float num){
  return num == (u64)num;
}

static inline unsigned formatF32(float num, char *str){
  char tmp[32];
  if (isInteger(num)) {
    return snprintf(tmp, 32, "%lld", (u64)num);
  } else {
    return snprintf(tmp, 32, "%.6e", num);
  }
}

void vectorDump(Vector *pVec){
  switch (pVec->type) {
    case VECTOR_TYPE_F32:
      vectorF32Dump(pVec);
      break;
    default:
      assert(0);
  }
}

static void vectorDeserialize(
  sqlite3_context *context,
  Vector *v
){
  switch (v->type) {
    case VECTOR_TYPE_F32:
      vectorF32Deserialize(context, v);
      break;
    default:
      assert(0);
  }
}

static void vectorSerialize(
  sqlite3_context *context,
  Vector *v
){
  switch (v->type) {
    case VECTOR_TYPE_F32:
      vectorF32Serialize(context, v);
      break;
    default:
      assert(0);
  }
}

size_t vectorSerializeToBlob(Vector *p, unsigned char *blob, size_t blobSize){
  switch (p->type) {
    case VECTOR_TYPE_F32:
      return vectorF32SerializeToBlob(p, blob, blobSize);
      break;
    default:
      assert(0);
  }
  return 0;
}

size_t vectorDeserializeFromBlob(Vector *p, const unsigned char *blob, size_t blobSize){
  switch (p->type) {
    case VECTOR_TYPE_F32:
      return vectorF32DeserializeFromBlob(p, blob, blobSize);
      break;
    default:
      assert(0);
  }
  return 0;
}

/**************************************************************************
** Vector index cursor implementations
****************************************************************************/

/*
** A VectorIdxCursor is a special cursor to perform vector index lookups.
 */
struct VectorIdxCursor {
  sqlite3 *db;          /* Database connection */
  DiskAnnIndex *index;   /* DiskANN index on disk */
};

/**
** Parses a type string such as `VECTOR(3)` and returns the number of dimensions.
**
** Returns -1 if the type string is not a valid vector type.
**/
static int parseVectorDims(const char *zType){
  int dims = 0;
  if( sqlite3_strnicmp(zType, "VECTOR(", 7)==0 ){
    const char *z = zType + 7;
    while( *z && *z!=')' ){
      if( !sqlite3Isdigit(*z) ){
        return -1;
      }
      dims = dims*10 + (*z - '0');
      z++;
    }
    if( *z==0 ){
      return -1;
    }
  }
  return dims;
}

int vectorIndexCreate(Parse *pParse, Index *pIdx){
  char zIndexFile[SQLITE_MAX_PATHLEN];
  sqlite3 *db = pParse->db;
  const char *zDbPath;
  Column *pCol;
  Table *pTab;
  int nDims;
  int i;
  pTab = pIdx->pTable;
  if (pIdx->nKeyCol != 1) {
    sqlite3ErrorMsg(pParse, "Only single column vector indexes are supported");
    return -1;
  }
  pCol = &pTab->aCol[pIdx->aiColumn[0]];
  nDims = parseVectorDims(sqlite3ColumnType(pCol, ""));
  if( nDims<0 ){
    sqlite3ErrorMsg(pParse, "Invalid vector type");
    return -1;
  }
  // TODO: Are we taking the right database here? Think about ATTACH.
  zDbPath = sqlite3_db_filename(db, db->aDb[0].zDbSName);

  // TODO: We may want to use a name that is unique to the _index_.
  sqlite3_snprintf(sizeof(zIndexFile), zIndexFile, "%s-vectoridx-%s", zDbPath, pIdx->zName);

  return diskAnnCreateIndex(db, zIndexFile, nDims);
}

int vectorIndexInsert(
  VectorIdxCursor *pCur,
  const BtreePayload *pX
){
  struct sqlite3_value *rowid;
  struct sqlite3_value *vec;
  UnpackedRecord r;
  r.aMem = pX->aMem;
  r.nField = pX->nMem;
  assert( r.nField == 2 );
  vec = r.aMem + 0;
  assert( sqlite3_value_type(vec) == SQLITE_BLOB );
  rowid = r.aMem + 1;
  assert( sqlite3_value_type(rowid) == SQLITE_INTEGER );
  Vector v;
  vectorInitStatic(&v, VECTOR_TYPE_F32, sqlite3_value_blob(vec));
  diskAnnInsert(pCur->index, &v, sqlite3_value_int64(rowid));
  return 0;
}

int vectorIndexCursorInit(
  sqlite3 *db,
  VdbeCursor *pCsr,
  const char *zIndexName
){
  VectorIdxCursor *pCur;
  char zIndexFile[SQLITE_MAX_PATHLEN];
  const char *zDbPath;
  int rc;

  // TODO: We're taking the filename of the currently selected
  // database (think attach). I think it's what we want to do,
  // but let's verify.
  zDbPath = sqlite3_db_filename(db, db->aDb[pCsr->iDb].zDbSName);

  // TODO: We may want to use a name that is unique to the _index_.
  sqlite3_snprintf(sizeof(zIndexFile), zIndexFile, "%s-vectoridx-%s", zDbPath, zIndexName);

  // TODO: Where do we deallocate this?
  pCur = sqlite3DbMallocZero(db, sizeof(VectorIdxCursor));
  if( pCur == 0 ){
    return SQLITE_NOMEM_BKPT;
  }
  rc = diskAnnOpenIndex(db, zIndexFile, &pCur->index);
  if( rc!=SQLITE_OK ){
    return rc;
  }
  pCur->db = db;
  pCsr->uc.pVecIdx = pCur;
  return SQLITE_OK;
}

void vectorIndexCursorClose(sqlite3 *db, VdbeCursor *pCsr){
  VectorIdxCursor *pCur = pCsr->uc.pVecIdx;
  diskAnnCloseIndex(pCur->index);
  sqlite3DbFree(db, pCur);
}

/**************************************************************************
** SQL function implementations
****************************************************************************/

/*
** Implementation of vector(X) function.
*/
static void vectorFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  char *zErrMsg = 0;
  Vector *pVec;
  int rc;
  if( argc < 1 ){
    return;
  }
  pVec = vectorContextAlloc(context, VECTOR_TYPE_F32);
  if( pVec==NULL ){
    return;
  }
  rc = vectorParse(argv[0], pVec, &zErrMsg);
  if( rc<0 ){
    sqlite3_result_error(context, zErrMsg, -1);
    sqlite3_free(zErrMsg);
    goto out_free_vec;
  }
  vectorSerialize(context, pVec);
out_free_vec:
  vectorFree(pVec);
}

/*
** Implementation of vector_extract(X) function.
*/
static void vectorExtractFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  const unsigned char *zStr;
  char *zErrMsg = 0;
  Vector *pVec;
  unsigned i;

  if( argc < 1 ){
    return;
  }
  pVec = vectorContextAlloc(context, VECTOR_TYPE_F32);
  if( pVec==NULL ){
    return;
  }
  if( vectorParse(argv[0], pVec, &zErrMsg)<0 ){
    sqlite3_result_error(context, zErrMsg, -1);
    sqlite3_free(zErrMsg);
    goto out_free;
  }
  vectorDeserialize(context, pVec);
out_free:
  vectorFree(pVec);
}

/*
** Implementation of vector_distance_cos(X, Y) function.
*/
static void vectorDistanceCosFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  Vector *pVec1, *pVec2;
  char *zErrMsg = 0;
  if( argc < 2 ) {
    return;
  }
  pVec1 = vectorContextAlloc(context, VECTOR_TYPE_F32);
  if( pVec1==NULL ){
    return;
  }
  pVec2 = vectorContextAlloc(context, VECTOR_TYPE_F32);
  if( pVec2==NULL ){
    goto out_free_vec1;
  }
  if( vectorParse(argv[0], pVec1, &zErrMsg)<0 ){
    sqlite3_result_error(context, zErrMsg, -1);
    sqlite3_free(zErrMsg);
    goto out_free_vec2;
  }
  if( vectorParse(argv[1], pVec2, &zErrMsg)<0){
    sqlite3_result_error(context, zErrMsg, -1);
    sqlite3_free(zErrMsg);
    goto out_free_vec2;
  }
  if( pVec1->len != pVec2->len ){
    sqlite3_result_error(context, "vectors must have the same length", -1);
    goto out_free_vec2;
  }
  sqlite3_result_double(context, vectorDistanceCos(pVec1, pVec2));
out_free_vec2:
  vectorFree(pVec2);
out_free_vec1:
  vectorFree(pVec1);
}

/*
** Register vector functions.
*/
void sqlite3RegisterVectorFunctions(void){
 static FuncDef aVectorFuncs[] = {
    VECTOR_FUNCTION(vector_distance_cos,  2, 0, 0, vectorDistanceCosFunc),

    FUNCTION(vector,         1, 0, 0, vectorFunc),
    FUNCTION(vector_extract, 1, 0, 0, vectorExtractFunc),
  };
  sqlite3InsertBuiltinFuncs(aVectorFuncs, ArraySize(aVectorFuncs));
}
#endif /* !defined(SQLITE_OMIT_VECTOR) */
