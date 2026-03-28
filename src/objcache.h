// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_OBJCACHE_H
#define JL_OBJCACHE_H

#include <condition_variable>
#include <thread>


#include <llvm/ADT/FunctionExtras.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <lmdb.h>

using CompileFn = llvm::unique_function<std::unique_ptr<llvm::MemoryBuffer>()>;

class ObjCache {
public:
    ObjCache();
    std::unique_ptr<llvm::MemoryBuffer> get(llvm::Module &M, CompileFn Compile);

protected:
    void writerThread();

private:
    MDB_env *Env = nullptr;
    std::thread WriterThread;
    std::vector<std::pair<llvm::ModuleHash, std::unique_ptr<llvm::MemoryBuffer>>> ObjQueue;
    std::mutex QueueMutex;
    std::condition_variable QueueCond;
};

#endif // JL_OBJCACHE_H
