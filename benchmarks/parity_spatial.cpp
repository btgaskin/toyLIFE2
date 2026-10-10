// Diagnostic counter for separately generated, safe author spatial variants.
// The orchestration script records the source patches and pinned identity.
#include "patternsplugin.h"
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: counter variant start_pair stop_pair output");
    const int variant = std::stoi(argv[1]);
    const int first = std::stoi(argv[2]), last = std::stoi(argv[3]);
    if (variant < 0 || variant > 5) throw std::runtime_error("variant must be in [0,5]");
    PatternsPlugin model;
    const int n = model.prot_energies.size();
    if (first < 0 || first > last || last > n*n) throw std::runtime_error("invalid pair interval");
    std::vector<std::uint64_t> abundance(n, 0);
    for (const int protein : model.prot_gen) ++abundance[protein];
    std::array<std::uint64_t, 8192> histogram{};
    std::uint64_t total = 0;
    std::array<int, 64> spread{};
    for (int signature = 0; signature < 64; ++signature)
      for (int state = 0; state < 6; ++state)
        spread[signature] |= ((signature >> state) & 1) << (2*state);
    for (int pair = first; pair < last; ++pair) {
      const int a = pair/n, b = pair%n;
      std::array<int, 64> frequencies_a{}, frequencies_b{};
      for (int promoter = 0; promoter < 16; ++promoter) {
        auto response = variant == 0 ? model.logic_proms(promoter, a, b)
                        : variant < 3 ? model.logic_proms_fixed(promoter, a, b)
                                      : model.logic_proms_eligible(promoter, a, b);
        // Isolate the identity rule: two gene products with the same protein ID
        // are copies of one competitor, rather than two annihilating identities.
        if ((variant == 2 || variant >= 4) && a == b && model.dimer_from_two(a, b) < 0)
          response[3] = response[4] = response[5] = response[1];
        if (variant == 5) {
          const int ab = model.dimer_from_two(a, b);
          for (int extra = 0; extra < 2; ++extra) {
            const int repeated = extra == 0 ? a : b;
            const int other = extra == 0 ? b : a;
            const int d = ab >= 0 ? ab : model.dimer_from_two(repeated, repeated);
            const int free = ab >= 0 ? repeated : other;
            if (d < 0) continue;
            const double de = model.dim_bond_energy[d] + model.prot_energies[ab >= 0 ? a : repeated]
                              + model.prot_energies[ab >= 0 ? b : repeated];
            const double binding = model.dim_prom_energies[promoter][d];
            const double poly = model.polymerase[promoter];
            const double free_binding = model.prot_prom_energies[promoter][free];
            const double free_energy = free_binding + model.prot_energies[free];
            if (binding < 0 && poly < 0 && std::abs(binding + de - poly) < 1e-8
                && !(free_binding < 0 && free_energy < poly - 1e-8))
              response[4 + extra] = free > 0 && free_binding < 0 ? model.prot_pol[promoter][free] : 0;
          }
        }
        int signature = 0;
        for (int state = 0; state < 6; ++state) signature |= response[state] << state;
        ++frequencies_a[a == 0 ? 0 : signature];
        ++frequencies_b[b == 0 ? 0 : signature];
      }
      const int flag = model.dimer_from_two(a, b) >= 0 ? 4096 : 0;
      const std::uint64_t weight = abundance[a]*abundance[b];
      for (int sa = 0; sa < 64; ++sa) if (frequencies_a[sa])
        for (int sb = 0; sb < 64; ++sb) if (frequencies_b[sb]) {
          const int code = (spread[sa] << 1) | spread[sb] | flag;
          histogram[code] += weight * frequencies_a[sa] * frequencies_b[sb];
        }
      total += weight*256;
      if (pair % (n*256) == 0) std::cerr << "pair " << pair << '/' << last << '\n';
    }
    std::ofstream out(argv[4]);
    if (!out) throw std::runtime_error("cannot open output");
    out << "{\"variant\":" << variant << ",\"start_pair\":" << first
        << ",\"stop_pair\":" << last << ",\"total_genotypes\":" << total
        << ",\"histogram\":{";
    bool separator = false;
    for (int code = 0; code < 8192; ++code) if (histogram[code]) {
      if (separator) out << ',';
      out << '"' << code << "\":" << histogram[code];
      separator = true;
    }
    out << "}}\n";
    if (!out) throw std::runtime_error("cannot write output");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
