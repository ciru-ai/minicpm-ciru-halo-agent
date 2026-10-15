{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [ gcc git pkg-config xxd python3 zstd ];
  shellHook = ''
    export CIRU_HOST_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.libdrm pkgs.elfutils pkgs.numactl pkgs.openssl ]}"
  '';
}
