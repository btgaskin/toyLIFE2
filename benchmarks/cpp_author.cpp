// Benchmark adapter: upstream source files are compiled without modification.
#include "toy_plugin.h"
#include "patterns/patternsplugin.h"
#include <memory>
#include <unistd.h>
#include <bitset>
#include <iomanip>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6) throw std::runtime_error("usage: cpp_author regulatory|metabolic|spatial_descriptor INPUT REPEATS OUTPUT [LOOPS] (cwd upstream/toylife)");
    const std::string workload(argv[1]);
    if (workload != "regulatory" && workload != "metabolic" && workload != "spatial_descriptor")
      throw std::runtime_error("unsupported workload");
    const int repeats = std::stoi(argv[3]);
    const int loops = argc == 6 ? std::stoi(argv[5]) : 1;
    if (loops < 1) throw std::runtime_error("loops must be positive");
    if (repeats < 1) throw std::runtime_error("repeats must be positive");
    std::ifstream in(argv[2]);
    if (!in) throw std::runtime_error("cannot open input");
    std::vector<std::string> genotypes;
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream row(line);
      long long gene;
      std::string genotype;
      while (row >> gene) {
        if (gene < 0 || gene >= (1 << 20)) throw std::runtime_error("gene outside 20-bit range");
        genotype += std::bitset<20>(gene).to_string();
      }
      if (!row.eof()) throw std::runtime_error("noninteger input");
      if (!genotype.empty()) {
        if (workload == "spatial_descriptor" && genotype.size() != 40) throw std::runtime_error("spatial descriptor requires two genes");
        if (genotype.size() > 20 * 16) throw std::runtime_error("maximum 16 genes");
        genotypes.push_back(genotype);
      }
    }
    if (genotypes.empty()) throw std::runtime_error("empty input");
    auto init_start = std::chrono::steady_clock::now();
    std::unique_ptr<ToyPlugin> plugin;
    std::unique_ptr<PatternsPlugin> spatial;
    if (workload == "spatial_descriptor") {
      if (chdir("patterns") != 0) throw std::runtime_error("cannot enter author patterns directory");
      spatial.reset(new PatternsPlugin());
      if (chdir("..") != 0) throw std::runtime_error("cannot restore author directory");
    } else plugin.reset(new ToyPlugin()); // Table loading excluded from evaluation times.
    const double initialization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - init_start).count();
    const std::vector<std::string> environment = {"00010001", "00000011", "01101010", "10100101"};
    std::vector<std::vector<int>> outputs(genotypes.size());
    std::vector<std::vector<int>> first_outputs;
    std::vector<double> durations;
    const bool diagnostic = std::getenv("TOYLIFE_BENCH_DIAGNOSTIC") != nullptr;
    double first_seconds = 0;
    for (int repetition = -1; repetition < repeats; ++repetition) {
      auto start = std::chrono::steady_clock::now();
      for (int loop = 0; loop < (repetition < 0 ? 1 : loops); ++loop)
      for (std::size_t i = 0; i < genotypes.size(); ++i) {
        if (diagnostic) {
          std::cerr << "diagnostic repetition=" << repetition << " loop=" << loop << " row=" << i << " genes=";
          for (std::size_t g = 0; g < genotypes[i].size(); g += 20) {
            if (g) std::cerr << ',';
            std::cerr << std::bitset<20>(genotypes[i].substr(g, 20)).to_ulong();
          }
          std::cerr << std::endl;
        }
        outputs[i] = workload == "spatial_descriptor"
          ? spatial->str_logic_function(genotypes[i])
          : workload == "regulatory"
          ? plugin->regulatory_phenotype(genotypes[i])
          : plugin->metabolic_phenotype(genotypes[i], environment);
      }
      auto end = std::chrono::steady_clock::now();
      const double duration = std::chrono::duration<double>(end - start).count();
      if (repetition < 0) {
        first_seconds = duration;
        first_outputs = outputs;
      } else {
        durations.push_back(duration);
        if (outputs != first_outputs) throw std::runtime_error("outputs changed across repetitions");
      }
    }
    std::ofstream out(argv[4]);
    if (!out) throw std::runtime_error("cannot open output");
    out << std::setprecision(17) << "{\"workload\":\"" << workload
        << "\",\"initialization_seconds\":" << initialization_seconds
        << ",\"first_seconds\":" << first_seconds
        << ",\"loops\":" << loops << ",\"rows\":" << genotypes.size() << ",\"output\":[";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (i) out << ',';
      out << '[';
      for (std::size_t j = 0; j < outputs[i].size(); ++j) {
        if (j) out << ',';
        out << outputs[i][j];
      }
      out << ']';
    }
    out << "],\"durations_seconds\":[";
    for (std::size_t i = 0; i < durations.size(); ++i) {
      if (i) out << ',';
      out << durations[i];
    }
    out << "]}\n";
    if (!out) throw std::runtime_error("output write failed");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
