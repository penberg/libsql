/*
** 2024-03-23
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
** DiskANN for SQLite/libSQL.
**
** The algorithm is described in the following publications:
**
**   Suhas Jayaram Subramanya et al (2019). DiskANN: Fast Accurate Billion-point
**   Nearest Neighbor Search on a Single Node. In NeurIPS 2019.
**
**   Aditi Singh et al (2021). FreshDiskANN: A Fast and Accurate Graph-Based ANN
**   Index for Streaming Similarity Search. ArXiv.
**
**   Yu Pan et al (2023). LM-DiskANN: Low Memory Footprint in Disk-Native
**   Dynamic Graph-Based ANN Indexing. In IEEE BIGDATA 2023.
**
** Here is the (internal, non-API) interface between this module and the
** rest of the SQLite system:
**
**    diskAnnOpenIndex()       Open a vector index file and return a DiskAnnIndex object.
**    diskAnnCloseIndex()      Close a DiskAnnIndex object.
**    diskAnnSearch()          Find the k-nearest neighbours of a vector.
**    diskAnnInsert()          Insert a vector to the index.
*/
#ifndef SQLITE_OMIT_VECTOR
#include "sqliteInt.h"

#include "vectorInt.h"

/* Objects */
typedef struct DiskAnnHeader DiskAnnHeader;
typedef struct SearchContext SearchContext;
typedef struct VectorMetadata VectorMetadata;
typedef struct VectorNode VectorNode;

/* TODO: Make this configurable. */
#define DISKANN_DEFAULT_SEARCH_L 200
#define DISKANN_DEFAULT_INSERT_L 70

/**
** The block size in bytes.
**/
#define DISKANN_BLOCK_SIZE 65536

/**
** The bit shift to get the block size in bytes.
**/
#define DISKANN_BLOCK_SIZE_SHIFT 9

struct DiskAnnHeader {
  i64 nMagic;                        /* Magic number */
  unsigned short nBlockSize;         /* Block size */
  unsigned short nVectorType;        /* Vector type */
  unsigned short nVectorDims;        /* Number of vector dimensions */
  unsigned short similarityFunction; /* Similarity function */
  i64 padding;                       /* Padding */
  i64 firstFreeOffset;               /* First free offset */
};

struct DiskAnnIndex {
  sqlite3_file  *pFd;             /* File descriptor */
  DiskAnnHeader header;           /* Header */
  i64 nFileSize;                  /* File size */
};

struct VectorMetadata {
  Vector *vec;
  u64 id;
  u64 offset;
};

struct VectorNode {
  Vector *vec;
  u64 id;
  u64 offset;
  int refCnt;
  int visited;                    /* Is this node visited? */
  VectorNode *pNext;              /* Next node in the visited list */
  u64 nNeighbours;                /* Number of neighbours */
  VectorMetadata aNeighbours[];   /* Neighbours */
};

/**************************************************************************
** Utility routines for managing vector nodes
**************************************************************************/

static VectorNode *vectorNodeNew(u64 id, u16 nNeighbours){
  VectorNode *pNode;
  pNode = sqlite3_malloc(sizeof(VectorNode) + nNeighbours * sizeof(VectorMetadata));
  if( pNode ){
    pNode->vec = NULL;
    pNode->id = id;
    pNode->offset = 0;
    pNode->refCnt = 1;
    pNode->nNeighbours = nNeighbours;
    pNode->visited = 0;
    pNode->pNext = NULL;
  }
  return pNode;
}

static void vectorNodeFree(VectorNode *pNode){
  for( int i = 0; i < pNode->nNeighbours; i++ ){
    vectorFree(pNode->aNeighbours[i].vec);
  }
  vectorFree(pNode->vec);
  sqlite3_free(pNode);
}

static VectorNode* vectorNodeGet(VectorNode *pNode){
  pNode->refCnt++;
  return pNode;
}

static void vectorNodePut(VectorNode *pNode){
  assert( pNode->refCnt>0 );
  pNode->refCnt--;
  if( pNode->refCnt==0 ){
    vectorNodeFree(pNode);
  }
}

/**************************************************************************
** Utility routines for parsing the index file
**************************************************************************/

#define VECTOR_METADATA_SIZE    (sizeof(u64) + sizeof(u16))
#define NEIGHBOUR_METADATA_SIZE (sizeof(u64) + sizeof(u64))

static unsigned int blockSize(DiskAnnIndex *pIndex){
  return pIndex->header.nBlockSize << DISKANN_BLOCK_SIZE_SHIFT;
}

static unsigned int vectorSize(DiskAnnIndex *pIndex){
  return sizeof(u32) + pIndex->header.nVectorDims * vectorElemSize(VECTOR_TYPE_F32);
}

static int diskAnnMaxNeighbours(DiskAnnIndex *pIndex){
  unsigned int nNeighbourVectorSize;
  unsigned int maxNeighbours;
  unsigned int nVectorSize;
  unsigned int nBlockSize;
  nBlockSize = blockSize(pIndex);
  nVectorSize = vectorSize(pIndex);
  nNeighbourVectorSize = vectorSize(pIndex);
  maxNeighbours = (nBlockSize - nVectorSize - VECTOR_METADATA_SIZE) / (nNeighbourVectorSize + NEIGHBOUR_METADATA_SIZE);
  assert( maxNeigbours > 0);
  return maxNeighbours;

}

static int neighbourMetadataOffset(DiskAnnIndex *pIndex){
  unsigned int nNeighbourVectorSize;
  unsigned int maxNeighbours;
  unsigned int nVectorSize;
  unsigned int nBlockSize;
  nBlockSize = blockSize(pIndex);
  nVectorSize = vectorSize(pIndex);
  nNeighbourVectorSize = vectorSize(pIndex);
  maxNeighbours = (nBlockSize - nVectorSize - VECTOR_METADATA_SIZE) / (nNeighbourVectorSize + NEIGHBOUR_METADATA_SIZE);
  assert( maxNeigbours > 0);
  return nVectorSize + VECTOR_METADATA_SIZE + maxNeighbours * (nNeighbourVectorSize); 
}

static int diskAnnReadHeader(
  sqlite3_file *pFd,
  DiskAnnHeader *pHeader
){
  int rc;
  // TODO: endianess
  rc = sqlite3OsRead(pFd, pHeader, sizeof(DiskAnnHeader), 0);
  assert( rc!=SQLITE_IOERR_SHORT_READ );
  return rc;
}

static int diskAnnWriteHeader(
  sqlite3_file *pFd,
  DiskAnnHeader *pHeader
){
  int rc;
  // TODO: endianess
  rc = sqlite3OsWrite(pFd, pHeader, sizeof(DiskAnnHeader), 0);
  return rc;
}

static VectorNode *diskAnnReadVector(
  DiskAnnIndex *pIndex,
  u64 offset
){
  unsigned char blockData[DISKANN_BLOCK_SIZE];
  VectorNode *pNode;
  u16 nNeighbours;
  int off = 0;
  u64 id;
  int rc;
  if( offset==0 ){
    return NULL;
  }
  assert( offset < pIndex->nFileSize );
  rc = sqlite3OsRead(pIndex->pFd, blockData, DISKANN_BLOCK_SIZE, offset);
  if( rc != SQLITE_OK ){
    return NULL;
  }
  id = (u64) blockData[off+0]
    | (u64) blockData[off+1] << 8
    | (u64) blockData[off+2] << 16
    | (u64) blockData[off+3] << 24
    | (u64) blockData[off+4] << 32
    | (u64) blockData[off+5] << 40
    | (u64) blockData[off+6] << 48
    | (u64) blockData[off+7] << 56;
  off += sizeof(u64);
  nNeighbours = (u16) blockData[off+0] | (u16) blockData[off+1] << 8;
  off += sizeof(u16);
  pNode = vectorNodeNew(id, nNeighbours);
  if( pNode==NULL ){
    return NULL;
  }
  pNode->offset = offset;
  pNode->vec = vectorAlloc(pIndex->header.nVectorType, pIndex->header.nVectorDims);
  if( pNode->vec==NULL ){
    vectorNodePut(pNode);
    return NULL;
  }
  off += vectorDeserializeFromBlob(pNode->vec, blockData+off, DISKANN_BLOCK_SIZE);
  for( int i = 0; i < nNeighbours; i++ ){
    VectorMetadata *pMetadata = &pNode->aNeighbours[i];
    pMetadata->vec = vectorAlloc(pIndex->header.nVectorType, pIndex->header.nVectorDims);
    if( pMetadata->vec==NULL ){
      vectorNodePut(pNode);
      return NULL;
    }
    off += vectorDeserializeFromBlob(pMetadata->vec, blockData+off, DISKANN_BLOCK_SIZE);
  }
  off = neighbourMetadataOffset(pIndex);
  for( int i = 0; i < nNeighbours; i++ ){
    pNode->aNeighbours[i].id = (u64) blockData[off+0]
      | (u64) blockData[off+1] << 8
      | (u64) blockData[off+2] << 16
      | (u64) blockData[off+3] << 24
      | (u64) blockData[off+4] << 32
      | (u64) blockData[off+5] << 40
      | (u64) blockData[off+6] << 48
      | (u64) blockData[off+7] << 56;
    off += sizeof(u64);
    pNode->aNeighbours[i].offset = (u64) blockData[off+0]
      | (u64) blockData[off+1] << 8
      | (u64) blockData[off+2] << 16
      | (u64) blockData[off+3] << 24
      | (u64) blockData[off+4] << 32
      | (u64) blockData[off+5] << 40
      | (u64) blockData[off+6] << 48
      | (u64) blockData[off+7] << 56;
    off += sizeof(u64);
  }
  return pNode;
}

static int diskAnnWriteVector(
  DiskAnnIndex *pIndex,
  Vector *pVec,
  u64 id,
  Vector **aNeighbours,
  VectorMetadata *aNeighbourMetadata,
  int nNeighbours,
  u64 offset,
  u64 nBlockSize
){
  char blockData[DISKANN_BLOCK_SIZE]; // TODO: dynamic allocation
  int rc = SQLITE_OK;
  int off = 0;
  memset(blockData, 0, DISKANN_BLOCK_SIZE);
  /* ID */
  blockData[off++] = id;
  blockData[off++] = id >> 8;
  blockData[off++] = id >> 16;
  blockData[off++] = id >> 24;
  blockData[off++] = id >> 32;
  blockData[off++] = id >> 40;
  blockData[off++] = id >> 48;
  blockData[off++] = id >> 56;
  /* nNeighbours */
  blockData[off++] = nNeighbours;
  blockData[off++] = nNeighbours >> 8;
  off += vectorSerializeToBlob(pVec, (void*)blockData+off, DISKANN_BLOCK_SIZE);
  for (int i = 0; i < nNeighbours; i++) {
    off += vectorSerializeToBlob(aNeighbours[i], (void*)blockData+off, DISKANN_BLOCK_SIZE);
  }
  off = neighbourMetadataOffset(pIndex);
  for( int i = 0; i < nNeighbours; i++ ){
    blockData[off++] = aNeighbourMetadata[i].id;
    blockData[off++] = aNeighbourMetadata[i].id >> 8;
    blockData[off++] = aNeighbourMetadata[i].id >> 16;
    blockData[off++] = aNeighbourMetadata[i].id >> 24;
    blockData[off++] = aNeighbourMetadata[i].id >> 32;
    blockData[off++] = aNeighbourMetadata[i].id >> 40;
    blockData[off++] = aNeighbourMetadata[i].id >> 48;
    blockData[off++] = aNeighbourMetadata[i].id >> 56;
    blockData[off++] = aNeighbourMetadata[i].offset;
    blockData[off++] = aNeighbourMetadata[i].offset >> 8;
    blockData[off++] = aNeighbourMetadata[i].offset >> 16;
    blockData[off++] = aNeighbourMetadata[i].offset >> 24;
    blockData[off++] = aNeighbourMetadata[i].offset >> 32;
    blockData[off++] = aNeighbourMetadata[i].offset >> 40;
    blockData[off++] = aNeighbourMetadata[i].offset >> 48;
    blockData[off++] = aNeighbourMetadata[i].offset >> 56;
  }
  rc = sqlite3OsWrite(pIndex->pFd, blockData, nBlockSize, pIndex->nFileSize);
  if( rc != SQLITE_OK ){
    return -1;
  }
  return nBlockSize;
}

/**
** Updates on-disk vector with a new neighbour, pruning the neighbour list if needed.
**/
static int diskAnnUpdateVectorNeighbour(
  DiskAnnIndex *pIndex,
  VectorNode *pVec,
  VectorNode *pNodeToAdd,
  Vector *pVecToAdd
) {
  unsigned char blockData[DISKANN_BLOCK_SIZE];
  unsigned int maxNeighbours = diskAnnMaxNeighbours(pIndex);
  u16 nNeighbours;
  int off;
  int rc;
  if( pVec->offset==0 ){
    return -1;
  }
  assert( offset < pIndex->nFileSize );
  rc = sqlite3OsRead(pIndex->pFd, blockData, DISKANN_BLOCK_SIZE, pVec->offset);
  if( rc != SQLITE_OK ){
    return -1;
  }
  nNeighbours = (u16) blockData[8] | (u16) blockData[9] << 8;
  off = sizeof(u64) + sizeof(u16) + vectorSize(pIndex);
  int insertIdx = -1;
  double toAddDist = vectorDistanceCos(pVecToAdd, pVec->vec);
  for( int i = 0; i < nNeighbours; i++ ){
    Vector neighbour;
    vectorInitStatic(&neighbour, pIndex->header.nVectorType, blockData+off);
    float dist = vectorDistanceCos(&neighbour, pVec->vec);
    if( toAddDist < dist ){
      insertIdx = i;
      break;
    }
    off += vectorSize(pIndex);
  }
  if( nNeighbours<maxNeighbours ){
    if( insertIdx==-1 ){
      insertIdx = nNeighbours;
    }
    nNeighbours++;
  } else {
    /* If the node to insert is to be pruned, just bail out. */
    if( insertIdx==-1 ){
      return SQLITE_OK;
    }
  }
  /* Calculate how many neighbours need to move. */
  int nToMove = nNeighbours-insertIdx-1;

  /* Move the neighbours to the right to make room. */
  off = sizeof(u64) + sizeof(u16) + vectorSize(pIndex) + insertIdx * vectorSize(pIndex);
  memmove(blockData+off+vectorSize(pIndex), blockData+off, nToMove * vectorSize(pIndex));

  /* Insert new neighbour to the list. */
  off = sizeof(u64) + sizeof(u16) + vectorSize(pIndex) + insertIdx * vectorSize(pIndex);
  vectorSerializeToBlob(pVecToAdd, (void*)blockData+off, DISKANN_BLOCK_SIZE);

  off = neighbourMetadataOffset(pIndex) + insertIdx * NEIGHBOUR_METADATA_SIZE;

  /* Move the metadata to right to make room. */
  memmove(blockData+off+NEIGHBOUR_METADATA_SIZE, blockData+off, nToMove * NEIGHBOUR_METADATA_SIZE);

  /* Insert new metadata to the list */
  blockData[off++] = pNodeToAdd->id;
  blockData[off++] = pNodeToAdd->id >> 8;
  blockData[off++] = pNodeToAdd->id >> 16;
  blockData[off++] = pNodeToAdd->id >> 24;
  blockData[off++] = pNodeToAdd->id >> 32;
  blockData[off++] = pNodeToAdd->id >> 40;
  blockData[off++] = pNodeToAdd->id >> 48;
  blockData[off++] = pNodeToAdd->id >> 56;
  blockData[off++] = pNodeToAdd->offset;
  blockData[off++] = pNodeToAdd->offset >> 8;
  blockData[off++] = pNodeToAdd->offset >> 16;
  blockData[off++] = pNodeToAdd->offset >> 24;
  blockData[off++] = pNodeToAdd->offset >> 32;
  blockData[off++] = pNodeToAdd->offset >> 40;
  blockData[off++] = pNodeToAdd->offset >> 48;

  blockData[8] = nNeighbours;
  blockData[9] = nNeighbours >> 8;
  rc = sqlite3OsWrite(pIndex->pFd, blockData, DISKANN_BLOCK_SIZE, pVec->offset);
  return rc;
}

/**************************************************************************
** DiskANN search
**************************************************************************/

struct SearchContext {
  Vector *pQuery;
  VectorNode **aCandidates;
  unsigned int nCandidates;
  unsigned int maxCandidates;
  VectorNode *visitedList;
  unsigned int nUnvisited;
  int k;
};

static void initSearchContext(SearchContext *pCtx, Vector* pQuery, unsigned int maxCandidates){
  pCtx->pQuery = pQuery;
  pCtx->aCandidates = sqlite3_malloc(maxCandidates * sizeof(VectorNode));
  pCtx->nCandidates = 0;
  pCtx->maxCandidates = maxCandidates;
  pCtx->visitedList = NULL;
  pCtx->nUnvisited = 0;
}

static void deinitSearchContext(SearchContext *pCtx){
  VectorNode *pNode, *pNext;

  pNode = pCtx->visitedList;
  while( pNode!=NULL ){
    pNext = pNode->pNext;
    vectorNodePut(pNode);
    pNode = pNext;
  }
  sqlite3_free(pCtx->aCandidates);
}

static int isVisited(SearchContext *pCtx, u64 id){
  for( VectorNode *pNode = pCtx->visitedList; pNode!=NULL; pNode = pNode->pNext ){
    if( pNode->id==id ){
      return 1;
    }
  }
  return 0;
}

static void addCandidate(SearchContext *pCtx, VectorNode *pNode){
  int candidateIdx = -1;
  // TODO: replace the check with a better data structure
  for( int i = 0; i < pCtx->nCandidates; i++ ){
    if( pCtx->aCandidates[i]->id==pNode->id ){
      return;
    }
  }
  // Special-case insertion to empty candidate set to avoid the distance calculation.
  if( pCtx->nCandidates==0 ){
    pCtx->aCandidates[pCtx->nCandidates++] = vectorNodeGet(pNode);
    pCtx->nUnvisited++;
    return;
  }
  // Find the index of the candidate that is further away from the query
  // vector than the one we're inserting.
  float toInsertDist = vectorDistanceCos(pCtx->pQuery, pNode->vec);
  for( int n = 0; n < pCtx->nCandidates; n++ ){
    float distCandidate = vectorDistanceCos(pCtx->pQuery, pCtx->aCandidates[n]->vec);
    if( toInsertDist < distCandidate ){
      candidateIdx = n;
      break;
    }
  }
  // If there is space for the new candidate, insert it; otherwise replace an
  // existing one.
  if( pCtx->nCandidates < pCtx->maxCandidates ){
    pCtx->nCandidates++;
    if( candidateIdx==-1 ){
      candidateIdx = pCtx->nCandidates-1;
    }
  } else {
    if( candidateIdx==-1 ){
      return;
    }
    VectorNode *toReplace = pCtx->aCandidates[pCtx->nCandidates-1];
    if( !toReplace->visited ){
      pCtx->nUnvisited--;
      vectorNodePut(toReplace);
    }
  }
  // Shift the candidates to the right to make space for the new one.
  for( int i = pCtx->nCandidates-1; i > candidateIdx; i-- ){
    pCtx->aCandidates[i] = pCtx->aCandidates[i-1];
  }
  // Insert the new candidate.
  pCtx->aCandidates[candidateIdx] = vectorNodeGet(pNode);
  pCtx->nUnvisited++;
}

static VectorNode* findClosestCandidate(SearchContext *pCtx){
  VectorNode *pCurrCandidate = NULL;
  for (int i = 0; i < pCtx->nCandidates; i++) {
    if( !pCtx->aCandidates[i]->visited ){
      if( pCurrCandidate==NULL ){
        pCurrCandidate = pCtx->aCandidates[i];
        continue;
      }
      VectorNode *pNewCandidate = pCtx->aCandidates[i];
      float currentDist = vectorDistanceCos(pCtx->pQuery, pCurrCandidate->vec);
      float newDist = vectorDistanceCos(pCtx->pQuery, pNewCandidate->vec);
      if( newDist < currentDist ){
        pCurrCandidate = pNewCandidate;
      }
    }
  }
  return pCurrCandidate;
}

static void markAsVisited(SearchContext *pCtx, VectorNode *pNode){
  pNode->visited = 1;
  assert(pCtx->nUnvisited > 0);
  pCtx->nUnvisited--;
  pNode->pNext = pCtx->visitedList;
  pCtx->visitedList = pNode;
}

static int hasUnvisitedCandidates(SearchContext *pCtx){
  return pCtx->nUnvisited > 0;
}

static int diskannBlockCount(DiskAnnIndex *pIndex){
  return (pIndex->nFileSize / blockSize(pIndex)) - 1;
}

static int diskAnnSearchInternal(
  DiskAnnIndex *pIndex,
  SearchContext *pCtx
){
  VectorNode *start;
  u64 entryOffset;
  u32 randomBlock;
  int nBlocks;

  nBlocks = diskannBlockCount(pIndex);
  if( nBlocks==0 ){
    return -1;
  }
  sqlite3_randomness(4, &randomBlock);
  randomBlock %= nBlocks;
  entryOffset = randomBlock * blockSize(pIndex);
  start = diskAnnReadVector(pIndex, entryOffset);
  if( start==NULL ){
    return 0;
  }
  addCandidate(pCtx, start);
  while( hasUnvisitedCandidates(pCtx) ){
    VectorNode *candidate = findClosestCandidate(pCtx);
    assert( candidate!=NULL );
    markAsVisited(pCtx, candidate);
    for( int i = 0; i < candidate->nNeighbours; i++ ){
      if( isVisited(pCtx, candidate->aNeighbours[i].id) ){
        continue;
      }
      VectorNode *neighbour = diskAnnReadVector(pIndex, candidate->aNeighbours[i].offset);
      if( neighbour==NULL ){
        continue;
      }
      addCandidate(pCtx, neighbour);
      vectorNodePut(neighbour);
    }
  }
  vectorNodePut(start);
  return 0;
}

int diskAnnSearch(
  DiskAnnIndex *pIndex,
  Vector *pVec,
  unsigned int k,
  i64 *aIds
){
  SearchContext ctx;
  int nIds = 0;
  int rc;

  initSearchContext(&ctx, pVec, DISKANN_DEFAULT_SEARCH_L);
  rc = diskAnnSearchInternal(pIndex, &ctx);
  if( rc==0 ){
    for( int i = 0; i < ctx.nCandidates; i++ ){
      if( i < k ){
        aIds[nIds++] = ctx.aCandidates[i]->id;
      }
    }
  }
  deinitSearchContext(&ctx);
  return nIds;
}

/**************************************************************************
** DiskANN insertion
**************************************************************************/

int diskAnnInsert(
  DiskAnnIndex *pIndex,
  Vector *pVec,
  i64 id
){
  unsigned int maxNeighbours = diskAnnMaxNeighbours(pIndex);
  unsigned int nNeighbours = 0;
  unsigned int nBlockSize;
  Vector **aNeighbours;
  VectorMetadata *aNeighbourMetadata;
  unsigned int nWritten;
  int rc = SQLITE_OK;
  VectorNode *pNode;
  SearchContext ctx;

  pNode = vectorNodeNew(id, nNeighbours);
  if( pNode==NULL ){
    return SQLITE_NOMEM;
  }
  pNode->offset = pIndex->nFileSize; // TODO: ensure this does not change later
  aNeighbours = sqlite3_malloc(maxNeighbours * sizeof(Vector*));
  if( aNeighbours==NULL ){
    rc = SQLITE_NOMEM;
    goto out_free_node;
  }
  aNeighbourMetadata = sqlite3_malloc(maxNeighbours * sizeof(VectorMetadata));
  if( aNeighbourMetadata==NULL ){
    rc = SQLITE_NOMEM;
    goto out_free_neighbours;
  }
  initSearchContext(&ctx, pVec, DISKANN_DEFAULT_INSERT_L);
  diskAnnSearchInternal(pIndex, &ctx);
  for( VectorNode *pVisited = ctx.visitedList; pVisited!=NULL; pVisited = pVisited->pNext ){
    if( nNeighbours==maxNeighbours ){
      break;
    }
    aNeighbours[nNeighbours] = pVisited->vec;
    aNeighbourMetadata[nNeighbours].id = pVisited->id;
    aNeighbourMetadata[nNeighbours].offset = pVisited->offset;
    nNeighbours++;
  }
  for( VectorNode* pVisited = ctx.visitedList; pVisited!=NULL; pVisited = pVisited->pNext ){
    diskAnnUpdateVectorNeighbour(pIndex, pVisited, pNode, pVec);
  }

  nBlockSize = blockSize(pIndex);
  nWritten = diskAnnWriteVector(pIndex, pVec, id, aNeighbours, aNeighbourMetadata, nNeighbours, pNode->offset, nBlockSize);

  deinitSearchContext(&ctx);

  if( nWritten<0 ){
    rc = SQLITE_ERROR;
    goto out_free_metadata;
  }
  pIndex->nFileSize += nWritten;

out_free_metadata:
  sqlite3_free(aNeighbourMetadata);
out_free_neighbours:
  sqlite3_free(aNeighbours);
out_free_node:
  vectorNodePut(pNode);
  return rc;
}

/**************************************************************************
** DiskANN index file management
**************************************************************************/

static int diskAnnOpenIndexFile(
  sqlite3 *db,
  const char *zName,
  sqlite3_file **ppFd
){
  int rc;
  rc = sqlite3OsOpenMalloc(db->pVfs, zName, ppFd,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, &rc
  );
  // TODO: close the file and free the memory
  return rc;
}

int diskAnnCreateIndex(
  sqlite3 *db,
  const char *zName,
  unsigned int nDims
){
  DiskAnnIndex *pIndex;
  int rc = SQLITE_OK;
  /* Allocate memory */
  pIndex = sqlite3_malloc(sizeof(DiskAnnIndex));
  if( pIndex == NULL ){
    rc = SQLITE_NOMEM;
    goto err_free;
  }
  /* Open index file */
  rc = diskAnnOpenIndexFile(db, zName, &pIndex->pFd);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  /* If the index already exists, don't trash it.  */
  rc = sqlite3OsFileSize(pIndex->pFd, &pIndex->nFileSize);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  if( pIndex->nFileSize > 0 ){
    goto out;
  }
  /* Initialize header */
  pIndex->header.nMagic = 0x4e4e416b736944; /* 'DiskANN' */
  pIndex->header.nBlockSize = DISKANN_BLOCK_SIZE >> DISKANN_BLOCK_SIZE_SHIFT;
  pIndex->header.nVectorType = VECTOR_TYPE_F32;
  pIndex->header.nVectorDims = nDims;
  pIndex->header.similarityFunction = 0;
  pIndex->header.padding = 0;
  pIndex->header.firstFreeOffset = 0;
  rc = diskAnnWriteHeader(pIndex->pFd, &pIndex->header);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  pIndex->nFileSize = blockSize(pIndex);
  rc = sqlite3OsTruncate(pIndex->pFd, pIndex->nFileSize);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
out:
  diskAnnCloseIndex(pIndex);
  return SQLITE_OK;
err_free:
  sqlite3_free(pIndex);
  return rc;
}

int diskAnnOpenIndex(
  sqlite3 *db,                    /* Database connection */
  const char *zName,              /* Index name */
  DiskAnnIndex **ppIndex          /* OUT: Index */
){
  DiskAnnIndex *pIndex;
  int rc = SQLITE_OK;
  /* Allocate memory */
  pIndex = sqlite3_malloc(sizeof(DiskAnnIndex));
  if( pIndex == NULL ){
    rc = SQLITE_NOMEM;
    goto err_free;
  }
  /* Open index file */
  rc = diskAnnOpenIndexFile(db, zName, &pIndex->pFd);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  /* Probe file size */
  rc = sqlite3OsFileSize(pIndex->pFd, &pIndex->nFileSize);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  if( pIndex->nFileSize == 0 ){
    goto err_free;
  }
  /* Read header */
  rc = diskAnnReadHeader(pIndex->pFd, &pIndex->header);
  if( rc != SQLITE_OK ){
    goto err_free;
  }
  *ppIndex = pIndex;
  return SQLITE_OK;
err_free:
  sqlite3_free(pIndex);
  return rc;
}

void diskAnnCloseIndex(DiskAnnIndex *pIndex){
  sqlite3OsCloseFree(pIndex->pFd);
  sqlite3_free(pIndex);
}
#endif /* !defined(SQLITE_OMIT_VECTOR) */
